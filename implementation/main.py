# !!! tested on python version 3.6 !!!
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, LeakyReLU, Dropout, GaussianNoise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Some GAN training techniques were used from: https//github.com/soumith/ganhacks


class GAN:
    def __init__(self):

        self.batch_size = 32
        self.log_step = 50
        self.scaler = MinMaxScaler((-1, 1))
        self.data = self.get_data_banknotes()
        self.init_model()

        # Logging loss
        self.logs_loss = pd.DataFrame(columns=['d_train_r',  # real data from discriminator training
                                               'd_train_f',  # fake data from discriminator training
                                               'd_test_r',  # real data from discriminator testing
                                               'd_test_f',  # fake data from discriminator testing
                                               'a'  # data from GAN(adversarial) training
                                               ])

        # Logging accuracy
        self.logs_acc = pd.DataFrame(columns=['d_train_r', 'd_train_f', 'd_test_r', 'd_test_f', 'a'])

        # Logging generated rows
        self.results = pd.DataFrame(columns=['iteration','variance', 'skewness', 'curtosis', 'entropy', 'prediction'])

    def get_data_banknotes(self):
        """
        Get data from file
        :return:
        """
        names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        dataset = pd.read_csv('data/data_banknotes.csv', names=names)
        dataset = dataset.loc[dataset['class'] == 0].values  # only real banknotes, because fake ones will be generated
        X = dataset[:, :4]  # omitting last column, we already know it will be 0
        data = self.structure_data(X)
        return data

    def scale(self, X):
        return self.scaler.fit_transform(X)

    def descale(self, X):
        return self.scaler.inverse_transform(X)

    def structure_data(self, X):
        """
        Structure data
        :param X:
        :return:
        """
        data_subsets = {'normal': X, 'scaled': self.scale(X)}
        for subset, data in data_subsets.items():  # splitting each subset on train and test
            splited_data = train_test_split(data, test_size=0.3, shuffle=True)
            data_subsets.update({
                subset: {
                    'train': splited_data[0],
                    'test': splited_data[1]}
            })

        return data_subsets

    def init_discriminator(self):
        """
        Init trainable discriminator model. Will be used for training and testing itself outside connected GAN model.
        LeakyReLU activation function, Adam optimizer and Dropout are recommended in GAN papers
        """
        self.D = Sequential()
        self.D.add(Dense(16, input_dim=4))
        self.D.add(LeakyReLU())
        self.D.add(Dropout(0.3))
        self.D.add(Dense(16))
        self.D.add(LeakyReLU())
        self.D.add(Dense(16))
        self.D.add(LeakyReLU())
        self.D.add(Dense(1, activation='sigmoid'))
        self.D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def init_discriminator_G(self):
        """
        Init non-trainable discriminator model. Will be used for training generator inside connected GAN model.
        LeakyReLU activation function, Adam optimizer and Dropout are recommended in GAN papers
        """
        self.Dg = Sequential()
        self.Dg.add(Dense(16, input_dim=4))  # activation function: ganhacks
        self.Dg.add(LeakyReLU())
        self.Dg.add(Dropout(0.3))
        self.Dg.add(Dense(16))
        self.Dg.add(LeakyReLU())
        self.Dg.add(Dense(16))
        self.Dg.add(LeakyReLU())
        # activation function: ganhacks
        self.Dg.add(Dense(1, activation='sigmoid'))
        self.Dg.trainable = False
        self.Dg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def init_generator(self):
        """
        LeakyReLU activation function, Adam optimizer and Dropout are recommended in GAN papers for BOTH D and G
        """
        self.G = Sequential()
        self.G.add(Dense(16, input_dim=64))
        self.G.add(LeakyReLU())
        self.G.add(Dropout(0.3))
        self.G.add(Dense(16))
        self.G.add(LeakyReLU())
        self.G.add(GaussianNoise(0.1))
        self.G.add(Dense(16))
        self.G.add(LeakyReLU())
        self.G.add(Dense(4, activation='tanh'))
        self.G.compile(loss='binary_crossentropy', optimizer='adam')

    def init_model(self):
        """
        Connecting non trainable model with Generator. Initializing D.
        :return:
        """
        self.init_discriminator()
        self.init_discriminator_G()
        self.init_generator()
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.GAN.add(self.Dg)
        self.GAN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_adversarial_data(self, mode='train'):
        """
        Get data for adversarial training.
        """
        data = self.data['scaled'][mode].copy()
        np.random.shuffle(data)
        features_real = data[:int(self.batch_size / 2)]  # random rows with real data

        noise = np.random.uniform(-1.0, 1.0, size=[int(self.batch_size / 2), 64])  # random noise for generator
        features_fake = self.G.predict(noise)  # fake data
        y_real = np.zeros([int(self.batch_size / 2), 1])  # array of zeros for real rows labels
        y_fake = np.ones([int(self.batch_size / 2), 1])  # array of ones for fake rows labels
        return features_real, y_real, features_fake, y_fake

    def train(self, train_steps):
        try:
            for i in range(train_steps):
                # Training D
                xr, yr, xf, yf = self.get_adversarial_data()  # train D separately from G
                d_loss_r = self.D.train_on_batch(xr, yr)  # separating real and fake data is recommended
                d_loss_f = self.D.train_on_batch(xf, yf)

                # Training G
                # flipping the label before prediction will
                # not influence D prediction as here D is not trainable and is getting weights from trainable D
                y = np.zeros([int(self.batch_size / 2), 1])  # flipping labels is recommended
                self.Dg.set_weights(self.D.get_weights())  # Copying weights from trainable D
                noise = np.random.uniform(-1.0, 1.0, size=[int(self.batch_size / 2), 64])  # getting input noise for G
                a_loss = self.GAN.train_on_batch(noise, y)

                # Testing
                xr_t, yr_t, xf_t, yf_t = self.get_adversarial_data(mode='test')
                d_pred_r = self.D.predict_on_batch(xr_t)  # getting example predictions
                d_pred_f = self.D.predict_on_batch(xf_t)
                d_loss_r_t = self.D.test_on_batch(xr_t, yr_t)  # getting loss and acc
                d_loss_f_t = self.D.test_on_batch(xf_t, yf_t)

                # Logging important data
                self.log(locals())
        finally:
            """
            Plot and save data when finished.
            """
            self.plot()
            self.results.to_csv('results/results.csv', index=False)

    def plot(self):
        """
        Preparing for plotting, plotting and saving plots.
        """
        import matplotlib.pyplot as plt

        ax_loss = self.logs_loss.plot(linewidth=0.75, figsize=(20, 10))
        ax_loss.set_xlabel('iteration')
        ax_loss.set_ylabel('loss')
        fig = plt.gcf()
        fig.set_dpi(200)
        plt.legend(loc='upper right', framealpha=0, prop={'size': 'large'})
        fig.savefig('results/loss.png', dpi=200)

        ax_acc = self.logs_acc.plot(linewidth=0.75, figsize=(20, 10))
        ax_acc.set_xlabel('iteration')
        ax_acc.set_ylabel('accuracy')
        fig = plt.gcf()
        fig.set_dpi(200)
        plt.legend(loc='upper right', framealpha=0, prop={'size': 'large'})
        fig.savefig('results/acc.png', dpi=200)

        plt.show()

    def log(self, variables):
        """
        Logging and printing all the necessary data
        """
        r_rows = pd.DataFrame(self.descale(variables['xr_t']), columns=['variance', 'skewness', 'curtosis', 'entropy'])
        r_rows['prediction'] = variables['d_pred_r']
        f_rows = pd.DataFrame(self.descale(variables['xf_t']), columns=['variance', 'skewness', 'curtosis', 'entropy'])
        f_rows['prediction'] = variables['d_pred_f']
        f_rows['iteration'] = variables['i']
        self.logs_loss = self.logs_loss.append(pd.Series(  # logging loss
                [variables['d_loss_r'][0],
                 variables['d_loss_f'][0],
                 variables['d_loss_r_t'][0],
                 variables['d_loss_f_t'][0],
                 variables['a_loss'][0]], index=self.logs_loss.columns), ignore_index=True)
        self.logs_acc = self.logs_acc.append(pd.Series(  # logging acc
                [variables['d_loss_r'][1],
                 variables['d_loss_f'][1],
                 variables['d_loss_r_t'][1],
                 variables['d_loss_f_t'][1],
                 variables['a_loss'][1]], index=self.logs_loss.columns), ignore_index=True)
        self.results = self.results.append(f_rows, ignore_index=True, sort=False)  # logging generated data
        if self.log_step and variables['i'] % self.log_step == 0:  # print metrics every 'log_step' iteration
            # preparing strings for printing
            log_msg = f""" 
Batch {variables['i']}:
    D(training):  
        loss:
            real : {variables['d_loss_r'][0]:.4f}
            fake : {variables['d_loss_f'][0]:.4f}
        acc: 
            real: {variables['d_loss_r'][1]:.4f}
            fake: {variables['d_loss_f'][1]:.4f}

    D(testing):  
        loss:
            real : {variables['d_loss_r_t'][0]:.4f}
            fake : {variables['d_loss_f_t'][0]:.4f}
        acc: 
            real: {variables['d_loss_r_t'][1]:.4f}
            fake: {variables['d_loss_f_t'][1]:.4f}
            
    GAN:
        loss: {variables['a_loss'][0]:.4f}
        acc: {variables['a_loss'][1]:.4f}
                        """
            print(log_msg)
            np.set_printoptions(precision=5, linewidth=140, suppress=True)  # set how np.array will be printed
            predictions = f"""
Example results:
    Real rows:

{r_rows}

    Fake rows:

{f_rows}
"""
            print(predictions)


if __name__ == '__main__':
    GAN().train(1500)


# TODO: refactor
