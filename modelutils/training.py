class TrainModel:
    def __init__(self, model, train_cll, train_drug, train_label, loss_fn, optimizer, num_epochs):
        self.model = model
        self.train_cll = train_cll
        self.train_drug = train_drug
        self.train_label = train_label
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self):
        for n in range(self.num_epochs):
            y_pred = self.model(self.train_cll, self.train_drug)
            loss = self.loss_fn(y_pred, self.train_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.model
