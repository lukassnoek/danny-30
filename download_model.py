import io
import zipfile
import requests

url = 'test'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()