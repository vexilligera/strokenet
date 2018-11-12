import requests
import re
import time
import base64
import subprocess
from io import BytesIO
from PIL import Image

class Draw:
	def setUrl(self, url):
		self.url = url

	def setSize(self, width, height):
		self.width = width
		self.height = height
		api = self.url + '/setsize/%d/%d' % (width, height)
		r = requests.get(api)

	def setRadius(self, radius):
		api = self.url + '/setradius/%f' % (radius)
		r = requests.get(api)

	def setColor(self, color):
		api = self.url + '/setcolor/%f/%f/%f' % (color[0], color[1], color[2])
		r = requests.get(api)

	def stroke(self, array):
		api = self.url + '/stroke'
		r = requests.post(api, json=array)

	def getImage(self):
		api = self.url + '/getimage'
		r = requests.get(api)
		data = re.sub('^data:image/.+;base64,', '', r.text)
		return Image.open(BytesIO(base64.b64decode(data)))

	def close(self):
		api = self.url + '/close'
		r = requests.get(api)
