class TrainModel:
    def __init__(self, model, train_feat, loss_fn, optimizer, num_epochs):
        self.model = model
        self.train_feat = train_feat
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self):
        for n in range(self.num_epochs):
            for train_X, train_label in self.train_feat:
                train_cll = train_X[0]
                train_drug = train_X[1]
                y_pred = self.model(train_cll, train_drug)
                loss = self.loss_fn(y_pred, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model
