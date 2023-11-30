import torch.nn as nn
import torch
import torch.nn.functional as F
class InfoNCEloss(nn.Module):
    def __init__(self, choose_neg, temperature):
        super(InfoNCEloss, self).__init__()
        self.neg = choose_neg
        self.temperature = temperature

    def forward(self, img):
        qk01 = img[0] * img[1] * 0.35
        qk02 = img[0] * img[2] * 0.35
        qk03 = img[0] * img[3] * 0.3
        qk_p = torch.sum(qk01 + qk02 + qk03, dim=-1)
        qk_n = torch.sum(img[0] * img[4:], dim=-1)

        logits = torch.cat([qk_p.view(1), qk_n], dim=0).unsqueeze(dim=0)
        labels = torch.zeros(1, dtype=torch.long, device=img.device)
        return F.cross_entropy(logits / self.temperature, labels)



class HardTripletloss(nn.Module):
    def __init__(self, args):
        super(HardTripletloss, self).__init__()
        self.m = args.margin
        self.num_topk1 = args.num_topk_pos
        self.num_topk2 = args.num_topk2_neg
        self.device = args.device

    def forward(self, img):
        D_p = 1 - torch.cosine_similarity(img[0], img[1:17], dim=-1)
        D_n = 1 - torch.cosine_similarity(img[0], img[17:], dim=-1)

        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        top_p, _ = D_p.topk(self.num_topk1, dim=0, largest=True)
        top_n, _ = D_n.topk(self.num_topk2, dim=0, largest=False)
        mean_p = torch.mean(top_p)
        for k in range(len(top_n)):
            if mean_p - top_n[k] + self.m <= 0:
                top_n[k] = top_n[k] * 0.0
            else:
                top_n[k] = mean_p - top_n[k] + self.m

        loss = torch.mean(top_n)
        return loss
