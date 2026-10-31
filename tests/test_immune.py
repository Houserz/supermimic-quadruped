"""Get-up training (plan D22): immune episodes keep rolling out after a fall, and every learner that
models termination is told so (world-model gate, critic input)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.policy.nets import Critic, CriticCfg, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import IMMUNE_LOGIT, WorldModel, WorldModelCfg

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def _tiny_wm():
    torch.manual_seed(0)
    heads = SimpleNamespace(termination_features=AnalyticHeads.termination_features)
    return WorldModel(WorldModelCfg(hidden=(32,), term_hidden=(16,)), heads, "cpu")


def _window(k=4, n=64):
    s = torch.randn(k + 1, n, 71)
    s[..., 3:7] = s[..., 3:7] / s[..., 3:7].norm(dim=-1, keepdim=True)
    refs = []
    for _ in range(k):
        q = torch.randn(n, 4)
        refs.append(ReferenceFrame(None, None, None, None, torch.randn(n, 3), q / q.norm(dim=-1, keepdim=True)))
    done = (torch.rand(k, n) < 0.3).float()
    return s, torch.randn(k, n, 29), torch.zeros(n, 33), done, refs


def test_termination_logits_gate():
    wm = _tiny_wm()
    s, _, _, _, refs = _window()
    immune = torch.arange(64) % 3 == 0
    raw = wm.termination_logits(s[1], refs[0])
    gated = wm.termination_logits(s[1], refs[0], immune)
    assert torch.equal(gated[~immune], raw[~immune])
    assert (gated[immune] == IMMUNE_LOGIT).all()
    assert torch.sigmoid(gated[immune]).max() < 1e-12


def test_window_loss_ignores_immune_labels():
    wm = _tiny_wm()
    s, a, xi, done, refs = _window()
    immune = torch.arange(64) % 3 == 0
    _, m = wm.window_loss(s, a, xi, done, refs, immune)
    # Flipping the labels of immune rows changes nothing ...
    flipped = torch.where(immune[None], 1.0 - done, done)
    _, m2 = wm.window_loss(s, a, xi, flipped, refs, immune)
    assert torch.allclose(m["term_bce"], m2["term_bce"]) and torch.allclose(m["term_acc"], m2["term_acc"])
    # ... and the termination loss equals the one on the non-immune rows alone.
    live = ~immune
    refs_live = [ReferenceFrame(None, None, None, None, r.root_pos_w[live], r.root_quat_w[live]) for r in refs]
    _, m3 = wm.window_loss(s[:, live], a[:, live], xi[live], done[:, live], refs_live)
    assert torch.allclose(m["term_bce"], m3["term_bce"], atol=1e-6)
    assert torch.allclose(m["term_acc"], m3["term_acc"], atol=1e-6)


def test_critic_upgrades_pre_immune_checkpoint():
    pcfg = PolicyCfg(goal_dim=680)
    new = Critic(CriticCfg(), pcfg=pcfg)
    with torch.no_grad():
        new.mlp.net[0].weight[:, -1] = 0.0
    old_sd = {k: v.clone() for k, v in new.state_dict().items()}
    old_sd["mlp.net.0.weight"] = old_sd["mlp.net.0.weight"][:, :-1]
    assert Critic.upgrade_state_dict(old_sd)
    assert not Critic.upgrade_state_dict(old_sd)  # idempotent
    loaded = Critic(CriticCfg(), pcfg=pcfg)
    loaded.load_state_dict(old_sd)
    x = (torch.randn(8, 71), torch.randn(8, 680), torch.randn(8, 33), torch.randn(8, 4))
    y = new(*x, torch.zeros(8, 1))
    torch.testing.assert_close(loaded(*x, torch.zeros(8, 1)), y)
    torch.testing.assert_close(loaded(*x, torch.ones(8, 1)), y)  # zero column: same value until trained


def test_windows_never_span_a_push():
    from supermimic.data.replay_buffer import ReplayBuffer

    E, T, k = 2, 40, 4
    buf = ReplayBuffer(E, T, 71, 29, 33, "cpu")
    ids = torch.arange(E)
    for t in range(30):
        cut = torch.tensor([t == 10, False])
        z = torch.zeros(E)
        buf.add(ids, torch.zeros(E, 71), torch.zeros(E, 29), torch.zeros(E, 29), torch.zeros(E, 33), z, z, torch.zeros(E, dtype=torch.long), torch.zeros(E, dtype=torch.long), torch.zeros(E, dtype=torch.bool), torch.zeros(E, dtype=torch.bool), cut=cut)
    m = buf.window_mask(k)
    # Env 0: the transition out of row 10 is a push, so no window may start at rows 7..10.
    assert not m[0, 7:11].any() and m[0, 6] and m[0, 11]
    assert m[1, 6:11].all()


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_lying_starts_land_on_the_floor():
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1"])
    n = 256
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=n, replay_capacity=100, lie_probability=1.0, seed=5), "cuda:0")
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        assert cmd.lying.all() and cmd.immune.all()
        max_speed = 0.0
        for _ in range(25):  # 0.5 s: drop and settle with a passive (default-pose) policy
            obs, _, term, _, _ = env.step(torch.zeros(n, 29, device=env.device))
            assert not term.any()
            max_speed = max(max_speed, float(robot.data.root_link_lin_vel_w.norm(dim=-1).max()))
        z = cmd.robot_root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        assert z.max() < 0.35, z.max()  # everyone is on the floor
        assert max_speed < 4.0, max_speed  # dropped, not launched by an interpenetrating start
        # All three pose families occur: the pelvis z axis points down/up (face down/up) or sideways.
        up = quat_apply_inverse(cmd.robot_root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
        assert (up[:, 0] < -0.7).any() and (up[:, 0] > 0.7).any() and (up[:, 1].abs() > 0.7).any()
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_pushes_knock_down_and_cut_windows():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1"])
    n = 64
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=n, replay_capacity=300, immune_probability=1.0, push_interval_s=(1.0, 2.0), seed=4), "cuda:0")
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term("motion")
        rec = env.recorder_manager.get_term("replay")
        down = torch.zeros(n, dtype=torch.bool, device=env.device)
        for _ in range(200):
            obs, *_ = env.step(torch.zeros(n, 29, device=env.device))  # default pose: stands unless pushed
            dz = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs()
            down |= dz > 0.25
        assert down.float().mean() > 0.5, f"only {down.float().mean():.0%} knocked down"
        buf = rec.buffer
        n_cut = int((buf.cut & buf.valid).sum())
        assert n_cut >= n, n_cut  # at least one push per env in 4 s
        m = buf.window_mask(8)
        for j in range(8):
            assert not (m & torch.roll(buf.cut, -j, dims=1)).any()
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_immune_env_falls_without_terminating():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    n = 64
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=n, replay_capacity=300, immune_probability=0.5, seed=3), "cuda:0")
    try:
        env.reset()
        cmd = env.command_manager.get_term("motion")
        tm = env.termination_manager
        rec = env.recorder_manager.get_term("replay")
        frac = []
        fallen_immune = torch.zeros(n, dtype=torch.bool, device=env.device)
        max_nefc = 0
        for _ in range(250):
            im = cmd.immune.clone()
            frac.append(float(im.float().mean()))
            env.step(1.5 * torch.randn(n, 29, device=env.device))  # violent actions: most robots fall
            fell = (tm.get_term("root_height") | tm.get_term("root_tilt"))
            assert not (fell & im).any(), "an immune episode was terminated by a root term"
            dz = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs()
            fallen_immune |= im & cmd.immune & (dz > 0.4)
            max_nefc = max(max_nefc, int(env.sim.data.nefc[:].max()))
        # Quota sampling keeps the live fraction on target although immune episodes outlast the others.
        assert abs(sum(frac) / len(frac) - 0.5) < 0.1, sum(frac) / len(frac)
        assert fallen_immune.sum() >= 3, "no immune robot ended up on the ground"
        assert max_nefc < 250, f"constraint rows {max_nefc} reach njmax: lying robots would lose contacts"
        buf = rec.buffer
        v = buf.valid
        assert buf.immune[v].any() and (~buf.immune[v]).any()
        # Immune episodes never carry a positive termination label.
        assert buf.done[v & buf.immune].sum() == 0
        w = buf.sample_windows(512, 8)
        assert w.immune.dtype == torch.bool and w.immune.shape == (512,)
        s = buf.sample_starts(512, hist=10)
        assert s.immune.any()
    finally:
        env.close()
