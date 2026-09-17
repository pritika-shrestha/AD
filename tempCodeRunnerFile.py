def __init__(self, cnn_path, svm_path, scaler_path):
        # Load CNN model
        self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)

        # Feature extractor: output of Dense(256) layer — layers[-3]
        # This matches exactly what was used during SVM training in the Colab notebook:
        #   feature_extractor = tf.keras.Model(
        #       inputs=cnn_model.layers[0].input,
        #       outputs=cnn_model.layers[-3].output   # Dense(256), before Dropout
        #   )
        self.feature_extractor = tf.keras.Model(
            inputs=self.cnn_model.layers[0].input,
            outputs=self.cnn_model.layers[-3].output
        )

        # Load SVM + scaler
        self.svm = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)

        # Categories — MUST match sorted(os.listdir(train_dir)) used during training.
        # Alphabetical sort of the 4 class folder names produces this order:
        #   0: Mild Dementia
        #   1: Moderate Dementia
        #   2: Non Demented
        #   3: Very mild Dementia
        self.CATEGORIES = [
            'Mild Dementia',
            'Moderate Dementia',
            'Non Demented',
            'Very mild Dementia',
        ]