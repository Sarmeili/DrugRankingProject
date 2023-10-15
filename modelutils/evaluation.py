class EvaluateModel:
    def __init__(self, model, test_cll, test_drug, test_label, loss_fn):
        self.model = model
        self.test_cll = test_cll
        self.test_drug = test_drug
        self.test_label = test_label
        self.loss_fn = loss_fn

    def evaluate(self):
        self.model.eval()
        y_pred_test = self.model(self.test_cll, self.test_drug)
        test_loss = self.loss_fn(y_pred_test, self.test_label)

        return test_loss
