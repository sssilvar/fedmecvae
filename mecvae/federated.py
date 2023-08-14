import copy
import torch
import pandas as pd

class FederatedTrainer:
    def __init__(self, model, criterion, optimizer_fn, learning_rate, n_clients, patience=3):
        self.n_clients = n_clients
        self.client_models = self._replicate_model(model, n_clients)
        self.client_optimizers = self._create_optimizers(self.client_models, optimizer_fn, learning_rate)
        self.criterion = criterion
        self.patience = patience
        self.train_losses = pd.DataFrame()
        self.test_losses = pd.DataFrame()

    def _replicate_model(self, model, n_clients):
        client_models = []
        for _ in range(n_clients):
            client_models.append(copy.deepcopy(model))
        return client_models

    def _create_optimizers(self, models, optimizer_fn, learning_rate):
        optimizers = []
        for model in models:
            optimizers.append(optimizer_fn(model.parameters(), lr=learning_rate))
        return optimizers

    def _train_client_model(self, client_model, client_optimizer, train_loader, grad_steps='auto'):
        client_model.train()
        total_loss = 0.0
        steps = 0

        for inputs, labels in train_loader:
            client_optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            client_optimizer.step()

            total_loss += loss.item()
            steps += 1

            if grad_steps != 'auto' and steps >= grad_steps:
                break

        return total_loss / steps

    def _evaluate(self, model, data_loader):
        model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                steps += 1

        return total_loss / steps

    def train(self, train_loaders, test_loader, grad_steps='auto'):
        best_loss = float('inf')
        patience_counter = 0

        for client_model, client_optimizer, train_loader in zip(
            self.client_models, self.client_optimizers, train_loaders
        ):
            num_samples = len(train_loader.dataset)
            batch_size = train_loader.batch_size

            if grad_steps == 'auto':
                grad_steps = num_samples // batch_size

            for step in range(grad_steps):
                client_model.train()
                inputs, labels = next(iter(train_loader))
                client_optimizer.zero_grad()
                outputs = client_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                client_optimizer.step()

                if (step + 1) % grad_steps == 0:
                    epoch = (step + 1) // grad_steps

                    train_loss = self._train_client_model(client_model, client_optimizer, train_loader, grad_steps=1)
                    test_loss = self._evaluate(client_model, test_loader)

                    self.train_losses = self.train_losses.append({'Client': client_model, 'Epoch': epoch,
                                                                  'Loss': train_loss}, ignore_index=True)
                    self.test_losses = self.test_losses.append({'Client': client_model, 'Epoch': epoch,
                                                                'Loss': test_loss}, ignore_index=True)

                    if test_loss < best_loss:
                        best_loss = test_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        break

    def federated_averaging(self):
        num_clients = len(self.client_models)
        model_params = self.client_models[0].state_dict()

        for param_name in model_params.keys():
            averaged_param = torch.zeros_like(model_params[param_name])
            for client_model in self.client_models:
                client_params = client_model.state_dict()
                averaged_param += client_params[param_name]
            averaged_param /= num_clients

            for client_model in self.client_models:
                client_params = client_model.state_dict()
                client_params[param_name] = averaged_param

        # Update all client models with the aggregated parameters
        for i in range(num_clients):
            self.client_models[i].load_state_dict(model_params)

    def optimize_rounds(self, train_loader, test_loader, num_rounds, grad_steps='auto'):
        for round in range(num_rounds):
            print(f"Round {round+1}/{num_rounds}")

            # Train the model using federated learning
            self.train(train_loader, test_loader, grad_steps=grad_steps)

            # Perform federated averaging
            self.federated_averaging()

    def get_aggregated_model(self):
        aggregated_model = self.client_models[0].__class__(**self.client_models[0].__dict__)
        aggregated_model.load_state_dict(self.client_models[0].state_dict())

        num_clients = len(self.client_models)
        model_params = aggregated_model.state_dict()

        for param_name in model_params.keys():
            averaged_param = torch.zeros_like(model_params[param_name])
            for client_model in self.client_models:
                client_params = client_model.state_dict()
                averaged_param += client_params[param_name]
            averaged_param /= num_clients

            model_params[param_name] = averaged_param

        aggregated_model.load_state_dict(model_params)
        return aggregated_model

    def get_train_losses(self):
        return self.train_losses

    def get_test_losses(self):
        return self.test_losses
