import credenciais
from oauth2client.client import GoogleCredentials
import googleapiclient.discovery

class Services_GCP():

    def __init__(self):
        self.credentials = credenciais.Credenciais()

    def GetService(self):
        credentials = GoogleCredentials.from_stream(self.credentials._credentials_file_path)
        service = googleapiclient.discovery.build("ml", "v1", credentials=credentials)
        return service

    def Response(self, dados):
        name = "projects/{}/models/{}".format(self.credentials._project_id, self.credentials._model_name)
        service = self.GetService()
        return service.projects().predict(name=name, body={"instances": dados}).execute()