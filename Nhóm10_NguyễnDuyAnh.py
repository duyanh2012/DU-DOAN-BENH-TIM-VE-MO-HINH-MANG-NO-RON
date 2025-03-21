import numpy as np
import pandas as pd
import heapq
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
file_path = "heart_disease_uci.csv"
data = pd.read_csv(file_path)

# Chọn các thuộc tính đầu vào và nhãn
features = ["age", "trestbps", "chol", "thalch", "oldpeak"]
label = "num"

X = data[features].values
y = data[label].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================== MẠNG NEURON NHÂN TẠO ========================== #
class OptimizedNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, reg_lambda=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.losses = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        error = y - output
        d_output = error * (output * (1 - output))
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.relu_derivative(self.z1)

        self.W2 += (self.a1.T.dot(d_output) * self.learning_rate) / m - self.reg_lambda * self.W2
        self.b2 += (np.sum(d_output, axis=0, keepdims=True) * self.learning_rate) / m
        self.W1 += (X.T.dot(d_hidden) * self.learning_rate) / m - self.reg_lambda * self.W1
        self.b1 += (np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate) / m

    def train(self, X, y, epochs=2000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Huấn luyện mô hình
nn = OptimizedNeuralNetwork(input_size=5, hidden_size=32, output_size=1, learning_rate=0.01, reg_lambda=0.001)
nn.train(X_train, y_train, epochs=2000)

# Đánh giá mô hình
y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n🎯 Độ chính xác: {accuracy:.4f}")
print(f"🔍 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n🧩 Ma trận nhầm lẫn:")
print(conf_matrix)

# Kiểm tra phân phối của nhãn
y_train_zeros = np.sum(y_train == 0)
y_train_ones = np.sum(y_train == 1)
print(f"Số lượng nhãn 0 trong tập huấn luyện: {y_train_zeros}")
print(f"Số lượng nhãn 1 trong tập huấn luyện: {y_train_ones}")

# ========================== THUẬT TOÁN A* ========================== #
def heuristic(x):
    return np.sum(x)

def a_star_search(X, top_n=3):
    queue = []
    for i in range(len(X)):
        heapq.heappush(queue, (-heuristic(X[i]), i, X[i]))
    top_patients = [heapq.heappop(queue) for _ in range(min(top_n, len(queue)))]
    return [(idx, patient_data) for _, idx, patient_data in top_patients]

high_risk_patients = a_star_search(X_test, top_n=3)
print("\n🔥 Các bệnh nhân có nguy cơ mắc bệnh cao nhất:")
for idx, patient in high_risk_patients:
    print(f"ID {idx}: {patient}")

# ========================== GIẢI THUẬT DI TRUYỀN ========================== #
def initialize_population(size, length):
    return [np.random.randn(length) for _ in range(size)]

def fitness_function(weights):
    return -np.mean(np.square(y_train - nn.forward(X_train)))

def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    return np.concatenate((parent1[:point], parent2[point:]))

def mutate(weights, rate=0.01):
    if random.random() < rate:
        idx = random.randint(0, len(weights) - 1)
        weights[idx] += np.random.randn()
    return weights

population_size = 20
chromosome_length = X_train.shape[1] * 32 + 32
population = initialize_population(population_size, chromosome_length)

for generation in range(50):
    population = sorted(population, key=fitness_function, reverse=True)
    new_population = []
    for i in range(0, population_size - 1, 2):
        if i + 1 < population_size:
            child1 = mutate(crossover(population[i], population[i+1]))
            child2 = mutate(crossover(population[i+1], population[i]))
            new_population.extend([child1, child2])
    population = new_population[:population_size]
best_weights = population[0]
print("\n🚀 Trọng số tối ưu từ Giải thuật di truyền:", best_weights[:5], "...")
