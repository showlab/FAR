import torch


class EMAModel:

    def __init__(self, model, decay):
        """
        Initialize the EMA object.

        Args:
        model (torch.nn.Module): The model to which EMA is applied.
        decay (float): The decay factor for EMA (typically between 0.99 and 0.999).
        """
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.original = None  # Stores the original parameters of the model

    def step(self, model):
        """
        Update the EMA shadow variables with the current model parameters.

        Args:
        model (torch.nn.Module): The current model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def store(self, model):
        """
        Store the original parameters of the model.
        """
        self.original = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    def restore(self, model):
        """
        Restore the original parameters of the model and clear the stored parameters.

        Args:
        model (torch.nn.Module): The model whose parameters are to be restored.

        Raises:
        RuntimeError: If the original parameters have not been stored.
        """
        if self.original is None:
            raise RuntimeError(
                'Original parameters have not been stored. Call `store()` before `restore()`.'
            )
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name].data)
        # Clear the stored original parameters
        self.original = None

    @torch.no_grad()
    def copy_to(self, model):
        """
        Apply the EMA shadow weights to the model.

        Args:
        model (torch.nn.Module): The target model.
        """
        model_params = []
        shadow_params = []
        for name, param in model.named_parameters():
            model_params.append(param.data)
            shadow_params.append(self.shadow[name].data.to(param.device))

        torch._foreach_copy_(model_params, shadow_params)

    def state_dict(self):
        """
        Return the model's state_dict for saving.

        Returns:
        dict: The state_dict of the model.
        """
        return {name: self.shadow[name].clone() for name in self.shadow}

    def load_state_dict(self, state_dict):
        """
        Load the state_dict into the EMA object.

        Args:
        state_dict (dict): The state_dict containing EMA weights to load.

        Raises:
        ValueError: If the keys in the state_dict do not match the current shadow keys.
        """
        if set(state_dict.keys()) != set(self.shadow.keys()):
            raise ValueError(
                'The provided state_dict does not match the structure of the EMA model.'
            )

        for name, param in state_dict.items():
            self.shadow[name].data.copy_(param)
