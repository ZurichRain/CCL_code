import torch
class AWP(object):
    """
    Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, loss_fn,optimizer, adv_param="weight", adv_lr=0.0001, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.criterion = loss_fn
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        _,y_preds = self.model(**inputs)

        adv_loss = self.criterion(y_preds.view(-1,4)[inputs['datas_mask'].view(-1) == 1], labels)
        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}