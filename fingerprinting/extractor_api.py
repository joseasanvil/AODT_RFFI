import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from dataset_preparation import ChannelIndSpectrogram
from deep_learning_models import NPairNet, identity_loss
from singleton import Singleton
from keras.models import load_model
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class ExtractorAPI(metaclass=Singleton):

    def train(self, data, label, dev_range, model_config, save_path = None):
        batch_size = model_config['batch_size']
        patience = model_config['patience']
        row = model_config['row']
        loss_type = model_config['loss_type']
        alpha = model_config['alpha']
        num_neg = model_config['loss_num_neg']
        npair_type = model_config['npair_type']
        
        ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        
        # Convert time-domain IQ samples to channel-independent spectrograms.
        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data, row)
        
        NPairNetObj = NPairNet()
        
        # Create an RFF extractor.
        feature_extractor = NPairNetObj.feature_extractor(data.shape)
        
        # Create the Triplet net using the RFF extractor.
        npair_net = NPairNetObj.create_npair_net(feature_extractor, alpha, num_neg, loss_type)

        # Create callbacks during training
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta = 0, patience = patience, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = patience, min_lr=1e-6, verbose=1)]
        
        # Split the dasetset into validation and training sets.
        data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.1, shuffle= True)
        del data, label
        
        # Create the trainining generator.
        train_generator = NPairNetObj.create_generator(batch_size, dev_range,  data_train, label_train, npair_type)
        # Create the validation generator.
        valid_generator = NPairNetObj.create_generator(batch_size, dev_range, data_valid, label_valid, npair_type)
        
        # Use the RMSprop optimizer for training.
        opt = RMSprop(learning_rate=1e-3)
        npair_net.compile(loss = identity_loss, optimizer = opt)

        # Start training.
        history = npair_net.fit(train_generator,
                                steps_per_epoch = data_train.shape[0]//batch_size,
                                epochs = 1000,
                                validation_data = valid_generator,
                                validation_steps = data_valid.shape[0]//batch_size,
                                verbose=1, 
                                callbacks = callbacks)

        if save_path:
            feature_extractor.save(save_path, overwrite=True)

        return feature_extractor, history

    def load(self, model_path, compile=False):
        return load_model(model_path, compile, safe_mode=False)

    def run(self, model, data, model_config):
        # Prepare input data for the model (convert to spectrogram images)
        data_freq = ChannelIndSpectrogram().channel_ind_spectrogram(data, model_config['row'])

        # Extract fingerprints from the trained model
        return model.predict(data_freq)

    def evaluate_closed_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, model_config, render_confusion_matrix=True):
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids == epoch_2_device_ids:
            print("Great! Epoch #1 and epoch #2 contain identical sets of device IDs. We can perform closed-set evaluation.")
        else:
            print("The device IDs in Epoch #2 and Epoch #1 must be identical. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.run(model, data_epoch_1, model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.run(model, data_epoch_2, model_config)
        labels_epoch_2_predicted = classifier.predict(fps_epoch_2)

        # Get the accuracy
        accuracy = accuracy_score(labels_epoch_2, labels_epoch_2_predicted)
        
        if render_confusion_matrix:
            conf_matrix = confusion_matrix(labels_epoch_2, labels_epoch_2_predicted)
            plt.figure(figsize=(12, 10), dpi=60)
            # TODO: sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', xticklabels=device_ids, yticklabels=device_ids)
            sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu')
            
            plt.title(f'Device Confusion Matrix (Euclidean Distance)')
            plt.xlabel('Device ID')
            plt.ylabel('Device ID')
            plt.tight_layout()
            plt.show()

        return accuracy
    
    def evaluate_open_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, model_config, render_roc_curve=True):
        # Here, we also expect two epochs. But we expect that the number set of devices in epoch #1 will be smaller compared to
        # the set of devices in epoch #2.
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids <= epoch_2_device_ids:
            print("Great! Epoch #2 contains more devices than #1, and #1 is a subset of #2. We can start open-set evaluation.")
        else:
            print("Device IDs in epoch #1 must be a subset of device IDs in epoch #2. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.run(model, data_epoch_1, model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.run(model, data_epoch_2, model_config)

        # Find the nearest 15 neighbors in the RFF database and calculate the distances to them.
        distances, indexes = classifier.kneighbors(fps_epoch_2)
        
        # Calculate the average distance to the nearest 15 neighbors.
        detection_score = distances.mean(axis =1)
  
        # Create a mask array which will contain 1 if device is from enrolled list, and 0 if it's new
        true_labels = [1 if item in epoch_1_device_ids else 0 for item in labels_epoch_2.flatten()]

        # Compute receiver operating characteristic (ROC).
        fpr, tpr, thresholds = roc_curve(true_labels, detection_score, pos_label = 1)

        # Invert false positive and true positive ratios to convert from distances to probabilities
        fpr = 1 - fpr  
        tpr = 1 - tpr

        # Compute EER
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        if render_roc_curve:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
            
            eer_point = min(zip(fpr, tpr), key=lambda x: abs(x[0] - (1-x[1])))
            plt.plot(eer_point[0], eer_point[1], 'ro', markersize=10, label=f'EER = {eer:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")