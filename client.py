import services_gcp as gcp
import numpy as np

lista = np.array([4.5,0,1,1,0,1,1,0,39.99])

# Dado de input para o modelo
dados = [{"input": lista.tolist()}]

# Conecta ao modelo
response = gcp.Services_GCP().Response(dados=dados)

if 'error' in response:
    raise RuntimeError(response['erro'])

result = response["predictions"][0]["totalvendas"][0]

print("Resultado: %.3f " % result )