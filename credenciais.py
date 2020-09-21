class Credenciais():

    # Default config
    PROJECT_ID = "tensorflow-app-290114"
    MODEL_NAME = "totalvendas"
    CREDENTIALS_FILE = "credentials/credentials.json"

    def __init__(self, project_id=PROJECT_ID, model_name=MODEL_NAME, credentials_file_path=CREDENTIALS_FILE):
        """Construtor da classe
           project_id: Id do projeto
           model_name: Nome do modelo dado no GCP (Google Cloud Platform)
           credentials_file_path: Caminho do arquivo de credenciais com extensão json
        """
        self._project_id = project_id
        self._model_name = model_name
        self._credentials_file_path = credentials_file_path

    def GetProjectId(self):
        return self._project_id

    def GetModelName(self):
        return self._model_name

    def GetCredentialsFilePath(self):
        return self._credentials_file_path