import torch


class Model1(torch.nn.Module):

    def __init__(self, cll_input_layer_size, drug_input_layer_size):
        super(Model1, self).__init__()
        self.cll_input_layer_size = cll_input_layer_size
        self.drug_input_layer_size = drug_input_layer_size

        self.linear1_ccl = torch.nn.Linear(self.cll_input_layer_size, int(self.cll_input_layer_size * 0.5))
        self.activation1_ccl = torch.nn.ReLU()

        self.linear2_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.5), int(self.cll_input_layer_size * 0.25))
        self.activation2_ccl = torch.nn.ReLU()

        self.linear3_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.25),
                                           int(self.cll_input_layer_size * 0.125))
        self.activation3_ccl = torch.nn.ReLU()

        self.linear4_ccl = torch.nn.Linear(int(self.cll_input_layer_size * 0.125),
                                           int(self.cll_input_layer_size * 0.0625))
        self.activation4_ccl = torch.nn.ReLU()

        self.linear1_drug = torch.nn.Linear(self.drug_input_layer_size, self.drug_input_layer_size - 100)
        self.activation1_drug = torch.nn.ReLU()

        self.linear2_drug = torch.nn.Linear(self.drug_input_layer_size - 100, self.drug_input_layer_size - 200)
        self.activation2_drug = torch.nn.ReLU()

        self.linear3_drug = torch.nn.Linear(self.drug_input_layer_size - 200,
                                            int((self.drug_input_layer_size - 200) * 0.5))
        self.activation3_drug = torch.nn.ReLU()

        self.linear1_concat = torch.nn.Linear(
            int(self.cll_input_layer_size * 0.0625) + int((self.drug_input_layer_size - 200) * 0.5), 1000)
        self.activation1_concat = torch.nn.ReLU()

        self.linear2_concat = torch.nn.Linear(1000, 500)
        self.activation2_concat = torch.nn.ReLU()

        self.linear3_concat = torch.nn.Linear(500, 250)
        self.activation3_concat = torch.nn.ReLU()

        self.linear4_concat = torch.nn.Linear(250, 1)
        self.mseloss = torch.nn.MSELoss()

    def forward(self, train_ccl, train_drug):
        train_ccl = self.linear1_ccl(train_ccl)
        train_ccl = self.activation1_ccl(train_ccl)
        train_ccl = self.linear2_ccl(train_ccl)
        train_ccl = self.activation2_ccl(train_ccl)
        train_ccl = self.linear3_ccl(train_ccl)
        train_ccl = self.activation3_ccl(train_ccl)
        train_ccl = self.linear4_ccl(train_ccl)
        train_ccl = self.activation4_ccl(train_ccl)

        train_drug = self.linear1_drug(train_drug)
        train_drug = self.activation1_drug(train_drug)
        train_drug = self.linear2_drug(train_drug)
        train_drug = self.activation2_drug(train_drug)
        train_drug = self.linear3_drug(train_drug)
        train_drug = self.activation3_drug(train_drug)

        concat_data = torch.cat((train_ccl, train_drug), 1)
        concat_data = self.linear1_concat(concat_data)
        concat_data = self.activation1_concat(concat_data)
        concat_data = self.linear2_concat(concat_data)
        concat_data = self.activation2_concat(concat_data)
        concat_data = self.linear3_concat(concat_data)
        concat_data = self.activation3_concat(concat_data)
        concat_data = self.linear4_concat(concat_data)
        # concat_data = self.mseloss(concat_data)

        return concat_data
