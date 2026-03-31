import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cubical_complex import CubicalComplex
from utils.PDMatching import SpatialAware_WassersteinDistance

class PDMatchingLoss(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        # Cubical complex constructor for persistent homology computation
        self.getPersistentInfo = CubicalComplex(dim=3)

        # distance between persistent diagrams
        #CORRECT THIS 
        self.criterion = SpatialAware_WassersteinDistance(p=p)
        self.tau = opt.tau
        self.alpha = opt.alpha
        # For precomputed ground truth persistent diagram.
        self.precal_PD = opt.precal_PD
        self.PD_target = {}
        self.pad_dims = (1, 1, 1, 1, 1, 1)

    def _pad_to_cube(self, x1, x2, D, H, W):
        tem_max = max(H, D, W)
        
        d_margin = abs(tem_max - D)
        h_margin = abs(tem_max - H)
        w_margin = abs(tem_max - W)

        d_left = d_margin // 2
        d_right = d_margin - d_left

        h_left = h_margin // 2
        h_right = h_margin - h_left

        w_left = w_margin // 2
        w_right = w_margin - w_left

        paddings = (
            w_left,
            w_right,
            h_left,
            h_right,
            d_left,
            d_right
        )

        if x1 is not None:
            x1 = F.pad(x1, paddings, mode="constant", value=0.0)

        if x2 is not None:
            x2 = F.pad(x2, paddings, mode="constant", value=0.0)

        return x1, x2

    def _pre_compute_PD(self, target, img_names):
        """
            Pre-compute PD for ground truth and save the training time.
        """
        # pad the boundary of the images by 1
        # padded_input = F.pad(target, self.pad_dims, mode='constant', value=1)
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        #channels are one for now, remove _ for further implementations later on
        N, _, D, H, W = target.size()

        # pad the image to square
        if H != W or W != D or D != H:
            _, padded_target = self._pad_to_cube(None, padded_target, D, H, W)

        padded_target = torch.clamp(padded_target, min=0.0, max=1.0)
        padded_target = 1.0 - padded_target

        for i in range(N):
            img = padded_target[i,0,:,:,:].unsqueeze(0).unsqueeze(0)
            self.PD_target[img_names[i]] = self.getPersistentInfo(img)

            # #DEBUG
            # print("img name:", img_names[i])
            # print("type:", type(self.PD_target[img_names[i]]))
            # print("len:", len(self.PD_target[img_names[i]]))
            # print("value:", self.PD_target[img_names[i]])
            break

    def forward(self, input, target, img_names=None):
        N, C, D, H, W = input.size()
        assert input.size() == target.size()
        assert input.device == target.device
        #NOTE TO SELF, CHANNELS CAN BE UPDATED FOR FUTURE PROJECTS, IE SUPPORT COLORMAPPING
        assert C == 1

        self.device = input.device

        input = input.to(torch.float32)
        target = target.to(torch.float32)

        # pad the boundary of the images by 1 (see Hu et al. NIPS 19' for reasons)
        padded_input = F.pad(input, self.pad_dims, mode='constant', value=1)
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, C, D, H, W = padded_input.size()

        # pad the image to square
        if H != W or W != D or D != H:
            padded_input, padded_target = self._pad_to_cube(padded_input, padded_target, D, H, W)

        padded_input = torch.clamp(padded_input, min=0.0, max=1.0)
        padded_target = torch.clamp(padded_target, min=0.0, max=1.0)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # invert the image color to fit the computation in CubicalComplex (super-level filtration)
        padded_input = 1.0 - padded_input
        padded_target = 1.0 - padded_target

        pi_x = self.getPersistentInfo(padded_input)
        if self.precal_PD:  # read ground truth persistent diagram from pre-computed
            pi_y = [self.PD_target[img_names[i]] for i in range(N)]            
        else:
            pi_y = self.getPersistentInfo(padded_target)

        for i in range(N):
            #H0 - connected componets
            pd_x_0 = pi_x[i][0][0]
            #punish more if > 2 connected compontes are detected, (edgecase for seperated GT optic nerve case)
            pd = pd_x_0.diagram
            pers = torch.abs(pd[:, 1] - pd[:, 0])
            num_meaningful = torch.sum(pers > self.tau)
            extra = torch.clamp(num_meaningful - 2, min=0)

            loss += self.alpha * extra.float()

            pd_y_0 = pi_y[i][0][0]

            # #H1 tunnels - loops 
            # pd_x_1 = pi_x[i][0][1]
            # pd_y_1 = pi_y[i][0][1]
            
            # #H2 cavities - voids
            # pd_x_2 = pi_x[i][0][2]
            # pd_y_2 = pi_y[i][0][2]

            # 2-nd persistant diagram ()
            wd_0 = self.criterion(pd_x_0, pd_y_0, D, H, W)
            # wd_1 = self.criterion(pd_x_1, pd_y_1, D, H, W)
            # wd_2 = self.criterion(pd_x_2, pd_y_2, D, H, W)

            loss += (wd_0)

        loss /= N

        return loss
