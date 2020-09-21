import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Definindo o nível do log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Desativando o Eager execution
tf.compat.v1.disable_eager_execution()

# Carregando os dados de treino e teste
train = pd.read_csv("datasets/vendas_data_training.csv", dtype=float)
test = pd.read_csv("datasets/vendas_data_test.csv", dtype=float)

# Separando as features e as labels dos dados de treino
x_train = train.drop("total_vendas", axis=1).values
y_train = train[["total_vendas"]].values

# Separando as features e as labels dos dados de teste
x_test = test.drop("total_vendas", axis=1).values
y_test = test[["total_vendas"]].values

# Operador de escala
scaler = MinMaxScaler(feature_range=(0, 1))

# Aplicando escala aos dados
x_scaled_train = scaler.fit_transform(x_train)
x_scaled_test = scaler.fit_transform(x_test)
y_scaled_train = scaler.fit_transform(y_train)
y_scaled_test = scaler.fit_transform(y_test)

# Definindo os hiperparâmetros
learning_rate = 0.001
epochs = 100
display_step = 5

# Definindo o número de inputs e outputs do modelo
num_inputs = 9
num_outputs = 1

# Definindo as camadas do modelo
layer_1_nodes = 100
layer_2_nodes = 200
layer_3_nodes = 100

# Tensorboard
RUN_NAME = "Layer 1: {0} - Layer 2: {1} - Layer 3: {2}".format(layer_1_nodes, layer_2_nodes, layer_3_nodes)

# Camada de input
with tf.compat.v1.variable_scope("input"):
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, num_inputs))

# Camada 1
with tf.compat.v1.variable_scope("layer1"):
    weights = tf.compat.v1.get_variable(name="weights1", shape=(num_inputs, layer_1_nodes), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None))
    biases = tf.compat.v1.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Camada 2
with tf.compat.v1.variable_scope("layer2"):
    weights = tf.compat.v1.get_variable(name="weights2", shape=(layer_1_nodes, layer_2_nodes), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None))
    biases = tf.compat.v1.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Camada 3
with tf.compat.v1.variable_scope("layer3"):
    weights = tf.compat.v1.get_variable(name="weights3", shape=(layer_2_nodes, layer_3_nodes), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None))
    biases = tf.compat.v1.get_variable(name="biases", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_outpout = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output
with tf.compat.v1.variable_scope("outpout"):
    weights = tf.compat.v1.get_variable(name="weights4", shape=(layer_3_nodes, num_outputs), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None))
    biases = tf.compat.v1.get_variable(name="biases4", shape=[num_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_outpout, weights) + biases

# Calculando a função de custo
with tf.compat.v1.variable_scope("cost"):
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, num_outputs))
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(prediction, Y))

# Aplicando o treinamento
with tf.compat.v1.variable_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

# Summary
with tf.compat.v1.variable_scope("logging"):
    tf.compat.v1.summary.scalar("current_cost", cost)
    summary = tf.compat.v1.summary.merge_all()

# Objeto para salvar o modelo
saver = tf.compat.v1.train.Saver()

# Criando a sessão tensorflow
with tf.compat.v1.Session() as session:

    # Restaurando o modelo
    saver.restore(session, "./modelos/modelo_treinado.ckpt")

    # FileWriter para tracking do progresso
    training_writer = tf.compat.v1.summary.FileWriter("./logs/{}/training".format(RUN_NAME), session.graph)
    test_writer = tf.compat.v1.summary.FileWriter("./logs/{}/test".format(RUN_NAME), session.graph)

    final_training_cost = session.run(cost, feed_dict={X: x_scaled_train, Y: y_scaled_train})
    final_test_cost = session.run(cost, feed_dict={X: x_scaled_test, Y: y_scaled_test})

    print("CUSTO EM TREINO: %.3f" % (final_training_cost))
    print("CUSTO EM TESTE: %.3f" % (final_test_cost))

    y_predicted_scaled = session.run(prediction, feed_dict={X: x_scaled_test})
    y_predicted = scaler.inverse_transform(y_predicted_scaled)

    real_total_sell = test["total_vendas"].values[0]
    predicted_total_sell = y_predicted[0][0]

    print("Valor real de 1 seguro: %.2f\nValor previsto de 1 seguro: %.2f" % (real_total_sell, predicted_total_sell))