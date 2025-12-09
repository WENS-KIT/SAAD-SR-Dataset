import torch
import torch.nn as nn
from ultralytics.models.yolo.detect.train import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.modules.detect_head_sr import DetectSR

# --------------------- Altitude Embedding ---------------------
class AltitudeEmbedding(nn.Module):
    def __init__(self, dim=1, out_channels=1024):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, altitude):
        if altitude is None:
            return None
        device = next(self.parameters()).device
        if isinstance(altitude, (float, int)):
            altitude = torch.tensor([[altitude]], dtype=torch.float32, device=device)
        elif isinstance(altitude, torch.Tensor):
            altitude = altitude.to(device)
            if altitude.dim() == 0:
                altitude = altitude.unsqueeze(0).unsqueeze(0)
            elif altitude.dim() == 1:
                altitude = altitude.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported altitude type: {type(altitude)}")

        embedded = self.embed(altitude)
        return embedded.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)

# --------------------- Altitude-Aware Detection Model ---------------------
class DetectionModelSR(DetectionModel):
    def __init__(self, cfg='yolov8sr.yaml', ch=3, nc=10, verbose=True):
        self.nc = nc
        ch_scalar = ch if isinstance(ch, int) else ch[0]
        super().__init__(cfg=cfg, ch=ch_scalar, nc=nc, verbose=verbose)
        self.altitude_embedding = AltitudeEmbedding(dim=1, out_channels=1024)
        if verbose:
            print(f"[INFO] DetectionModelSR initialized with {self.nc} classes")

    def init_criterion(self):
        self.criterion = v8DetectionLoss(self)

    def forward(self, x, altitude=None, augment=False, profile=False, visualize=False):
        outputs = []
        altitude_feat = None
        prediction_outputs = None
        feature_maps = []

        if altitude is not None:
            try:
                altitude_feat = self.altitude_embedding(altitude)
            except Exception as e:
                print(f"[WARNING] Altitude embedding failed: {e}")

        for i, m in enumerate(self.model):
            try:
                if hasattr(m, 'f'):  # Module has fused inputs
                    x_in = [outputs[j] if j != -1 else x for j in m.f] if isinstance(m.f, list) else [outputs[m.f] if m.f != -1 else x]
                    if isinstance(m, DetectSR):
                        prediction_outputs = m(x_in, altitude=altitude_feat)

                        # Reshape for loss
                        feature_maps = []
                        for o in prediction_outputs:
                            B, C, H, W = o.shape
                            na = m.na
                            no = m.no
                            assert C == na * no, f"[❌] Expected {na * no} channels but got {C}"
                            reshaped = o.view(B, na, no, H, W)                      # (B, 3, 74, H, W)
                            reshaped = reshaped.permute(0, 2, 3, 4, 1).contiguous()  # (B, 74, H, W, 3)
                            reshaped = reshaped.view(B, no, H, W * na)               # (B, 74, H, 240)
                            feature_maps.append(reshaped)

                        x = prediction_outputs
                    elif m.__class__.__name__ in ['Concat', 'Add']:
                        x = m(x_in)
                    else:
                        x = m(x_in[0] if len(x_in) == 1 else x_in)
                else:
                    x = m(x)

                outputs.append(x if isinstance(x, torch.Tensor) else x[0])

            except Exception as e:
                print(f"[ERROR] Forward failed at layer {i} ({m.__class__.__name__}): {e}")
                raise

        return prediction_outputs, feature_maps

    def predict(self, x, altitude=None, augment=False, visualize=False):
        with torch.no_grad():
            preds, _ = self.forward(x, altitude=altitude, augment=augment, visualize=visualize)
        return preds

    def get_model_info(self):
        return {
            'num_classes': self.nc,
            'has_altitude_embedding': hasattr(self, 'altitude_embedding'),
            'detect_layer_type': 'DetectSR'
        }

