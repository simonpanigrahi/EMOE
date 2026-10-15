import torch
import torch.nn as nn
import torch.nn.functional as F

class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        self.t = t
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1)))/self.t
        output = torch.softmax(x, dim=1)
        return output


# ---------------------------------------------------------------------------
# Alternatives to EMOE's router, for the router experiments.
# Every router returns a (batch, 3) tensor: one weight each for
# text, video (face) and audio (voice), in that order, summing to 1 per clip.
# ---------------------------------------------------------------------------

class LightRouter(nn.Module):
    """A small router that looks at each modality separately.

    EMOE's router flattens all 50 time steps of text, video and audio into one
    vector of 39,650 numbers (97% of them text) and needs 196.5M parameters.
    This one first averages each modality over time, gives each modality the
    same number of features (64), and only then decides the three weights.
    About 63K parameters.
    """

    def __init__(self, dims, temperature, hidden=64):
        super().__init__()
        self.dims = dims  # sizes of [text, video, audio], the order they are glued in
        # One small projection per modality, so each modality gets 64 features
        self.proj_text = nn.Linear(dims[0], hidden)
        self.proj_video = nn.Linear(dims[1], hidden)
        self.proj_audio = nn.Linear(dims[2], hidden)
        # Then the same two-layer shape as EMOE's router, just much narrower
        self.l1 = nn.Linear(3 * hidden, hidden)
        self.l2 = nn.Linear(hidden, 3)
        self.t = temperature

    def forward(self, x):
        # x: (batch, time steps, text + video + audio features), e.g. (16, 50, 793)
        text, video, audio = torch.split(x, self.dims, dim=2)
        # Average over the time steps: one summary vector per modality per clip
        summary = torch.cat([
            self.proj_text(text.mean(dim=1)),
            self.proj_video(video.mean(dim=1)),
            self.proj_audio(audio.mean(dim=1)),
        ], dim=1)  # (batch, 192)
        # Same steps as EMOE's router: linear, L2-normalize, ReLU, linear, temperature, softmax
        hidden = F.relu(F.normalize(self.l1(summary), p=2, dim=1))
        return torch.softmax(self.l2(hidden) / self.t, dim=1)


class EqualRouter(nn.Module):
    """No router at all: every clip gives each modality the same weight, 1/3."""

    def forward(self, x):
        return torch.full((x.shape[0], 3), 1.0 / 3, device=x.device)


class ErrorHead(nn.Module):
    """Guesses how wrong one expert's sentiment prediction will be.

    Input: the feature the expert uses for its own prediction.
    Output: a guessed absolute error, always >= 0 (Softplus). About 16K parameters.
    """

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, feature):
        return self.net(feature)  # (batch, 1)


def weights_from_error_guesses(err_text, err_video, err_audio):
    """Turn three guessed errors into three weights that sum to 1.

    Same formula EMOE uses to build its router's training target:
    weight = 1 / (error^2 + 0.1), then divide by the sum of the three.
    A small guessed error gives a big weight.
    """
    trust = torch.cat([
        1.0 / (err_text ** 2 + 0.1),
        1.0 / (err_video ** 2 + 0.1),
        1.0 / (err_audio ** 2 + 0.1),
    ], dim=1)  # (batch, 3)
    return trust / trust.sum(dim=1, keepdim=True)
