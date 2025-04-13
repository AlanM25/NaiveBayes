import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout,
    QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt

from model import NaiveBayesGolfModel
from data_processing import DataProcessing

class NaiveBayesGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naive Bayes Golf Predictor")
        self.setGeometry(100, 100, 1200, 700)

        self.df_original = None
        self.df_laplace = None
        self.model = None
        self.processor = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Botones
        btn_load = QPushButton("Cargar Dataset")
        btn_load.clicked.connect(self.load_dataset)
        layout.addWidget(btn_load)

        btn_train = QPushButton("Procesar y Entrenar")
        btn_train.clicked.connect(self.process_and_train)
        layout.addWidget(btn_train)

        btn_show_original = QPushButton("Mostrar Dataset Original")
        btn_show_original.clicked.connect(lambda: self.display_table(self.df_original, "Dataset Original"))
        layout.addWidget(btn_show_original)

        btn_show_laplace = QPushButton("Mostrar Dataset con Laplacian")
        btn_show_laplace.clicked.connect(lambda: self.display_table(self.df_laplace, "Dataset con Laplacian"))
        layout.addWidget(btn_show_laplace)

        btn_show_tables = QPushButton("Mostrar Tablas de Probabilidad")
        btn_show_tables.clicked.connect(self.show_probability_tables)
        layout.addWidget(btn_show_tables)

        btn_export = QPushButton("Exportar Predicciones a CSV")
        btn_export.clicked.connect(self.export_predictions)
        layout.addWidget(btn_export)

        # Tabla de dataset
        self.label_status = QLabel("Dataset no cargado")
        layout.addWidget(self.label_status)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Predicción nueva instancia
        layout.addWidget(QLabel("Seleccionar condición climática:"))
        self.combo = QComboBox()
        self.combo.addItems(["Sunny", "Overcast", "Rainy"])
        layout.addWidget(self.combo)

        btn_predict = QPushButton("Predecir si se jugará golf")
        btn_predict.clicked.connect(self.predict_new_instance)
        layout.addWidget(btn_predict)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir dataset", "", "CSV (*.csv);;TXT (*.txt);;Todos los archivos (*)")
        if file_path:
            try:
                self.df_original = pd.read_csv(file_path)
                self.label_status.setText(f"Archivo cargado: {file_path}")
                self.display_table(self.df_original)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")

    def display_table(self, df, title="Tabla"):
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        self.label_status.setText(title)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def process_and_train(self):
        if self.df_original is None:
            QMessageBox.warning(self, "Advertencia", "Primero carga un dataset.")
            return

        # Procesar
        df = DataProcessing.add_laplace_correction(self.df_original.copy())
        self.df_laplace = df
        self.processor = DataProcessing(df)
        outlook_encoded, play_encoded = self.processor.encode_features()

        self.model = NaiveBayesGolfModel()
        self.model.train_model(outlook_encoded, play_encoded)

        freq = self.model.create_frequency_table(df)
        like = self.model.create_likelihood_table(freq)
        post = self.model.create_posterior_table(like)

        print("Frecuencia:\n", freq)
        print("Likelihood:\n", like)
        print("Posterior:\n", post)

        QMessageBox.information(self, "Éxito", "Modelo entrenado y tablas generadas. Consulta la consola.")

    def predict_new_instance(self):
        if self.model is None or self.processor is None:
            QMessageBox.warning(self, "Advertencia", "Primero entrena el modelo.")
            return

        value = self.combo.currentText()
        le_outlook = self.processor.le_outlook
        le_play = self.processor.le_play

        condition_transformed = le_outlook.transform([value])
        yhat_prob = self.model.model.predict_proba([condition_transformed])
        prediction_statement = self.model.model.predict([condition_transformed])
        prediction = le_play.inverse_transform(prediction_statement)

        self.result_label.setText(f"Predicción: {prediction[0]} (Probabilidades: {yhat_prob[0]})")

    def show_probability_tables(self):
        if self.model is None or self.df_laplace is None:
            QMessageBox.warning(self, "Advertencia", "Primero entrena el modelo.")
            return

        freq = self.model.outlook_play_df
        like = self.model.outlook_likelihood_df
        post = self.model.posterior_probability

        # Mostrar por consola (ya está)
        print("Frecuencia:\n", freq)
        print("Likelihood:\n", like)
        print("Posterior:\n", post)

        # Mostrar en tabla principal (una por una)
        self.display_table(freq.reset_index(), "Tabla de Frecuencia")
        QMessageBox.information(self, "Tablas", "Mostrando tabla de frecuencia. Ahora tabla de likelihood.")
        self.display_table(like.reset_index(), "Tabla de Likelihood")
        QMessageBox.information(self, "Tablas", "Mostrando tabla de likelihood. Ahora tabla posterior.")
        self.display_table(post.reset_index(), "Tabla de Probabilidad Posterior")

    def export_predictions(self):
        if self.model is None or self.processor is None:
            QMessageBox.warning(self, "Advertencia", "Entrena primero el modelo.")
            return

        predictions = self.model.make_predictions(self.processor.le_outlook, self.processor.le_play)

        data = []
        for condition, prediction, yhat_prob in predictions:
            data.append({
                "Outlook": condition,
                "Prediction": prediction[0],
                "Prob_Yes": yhat_prob[0][1],
                "Prob_No": yhat_prob[0][0]
            })

        df_preds = pd.DataFrame(data)

        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "", "CSV files (*.csv)")
        if path:
            df_preds.to_csv(path, index=False)
            QMessageBox.information(self, "Exportado", f"Predicciones guardadas en {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NaiveBayesGUI()
    window.show()
    sys.exit(app.exec())
