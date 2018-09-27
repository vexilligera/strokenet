var Filter = function(layerTree) {
	this.layerTree = layerTree;
	this.mesh = null;
	this.rectMesh = null;
	this.glMeshArray = null;
	this.meshResolution = -1;
	this.texture = null;
	this.node = null;
};

Filter.prototype = {
	selectLayer: function(node) {
		this.node = node;
		var gl = this.layerTree.gl;
		if (this.texture != null)
			gl.deleteTexture(this.texture);
		if (this.node.selectedPixel != null) {
			this.texture = gl.createTexture();
			gl.bindTexture(gl.TEXTURE_2D, this.texture);
			node.setTextureParameter(false, true);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.node.areaWidth, this.node.areaHeight,
							0, gl.RGBA, gl.UNSIGNED_BYTE, this.node.selectedPixel);
		}
		else {

		}
	},

	/*
		boundingbox is a Float32Array(18) array
		*0---*1--2*
		|    |    |
		*3---*4---*5
		|    |    | each * records x and y
		*6---*7---*8
	*/
	createMesh: function(boundingBox, resolution) {
		if (!this.mesh)
			this.mesh = new Float32Array(resolution * resolution * 2);
		this.meshResolution = resolution;
		for (var k = 0; k < 4; ++k) {
			var x0, y0, x1, y1, x2, y2, x3, y3;
			switch (k) {
			case 0:
				x0 = boundingBox[8], y0 = boundingBox[9], x1 = boundingBox[10], y1 = boundingBox[11],
				x2 = boundingBox[2], y2 = boundingBox[3], x3 = boundingBox[4], y3 = boundingBox[5];
				break;
			case 1:
				x0 = boundingBox[6], y0 = boundingBox[7], x1 = boundingBox[8], y1 = boundingBox[9],
				x2 = boundingBox[0], y2 = boundingBox[1], x3 = boundingBox[2], y3 = boundingBox[3];
				break;
			case 2:
				x0 = boundingBox[12], y0 = boundingBox[13], x1 = boundingBox[14], y1 = boundingBox[15],
				x2 = boundingBox[6], y2 = boundingBox[7], x3 = boundingBox[8], y3 = boundingBox[9];
				break;
			case 3:
				x0 = boundingBox[14], y0 = boundingBox[15], x1 = boundingBox[16], y1 = boundingBox[17],
				x2 = boundingBox[8], y2 = boundingBox[9], x3 = boundingBox[10], y3 = boundingBox[11];
				break;
			default:
				break;
			}
			var halfRes = (resolution - 1) / 2;
			var startX = x0, startY = y0, endX = x1, endY = y1;
			var stepStartX = (x2 - x0) / halfRes;
			var stepStartY = (y2 - y0) / halfRes;
			var stepEndX = (x3 - x1) / halfRes;
			var stepEndY = (y3 - y1) / halfRes;
			for (var i = 0; i <= halfRes; ++i) {
				var x = startX, y = startY;
				var stepX = (endX - startX) / halfRes;
				var stepY = (endY - startY) / halfRes;
				for (var j = 0; j <= halfRes; ++j) {
					var x_index = j, y_index = i;
					switch (k) {
					case 0:
						x_index += halfRes, y_index += halfRes;
						break;
					case 1:
						y_index += halfRes;
						break;
					case 3:
						x_index += halfRes;
						break;
					default:
						break;
					}
					x_index = 2 * (x_index  + y_index * resolution);
					y_index = x_index + 1;
					this.mesh[x_index] = x, this.mesh[y_index] = y;
					x += stepX;
					y += stepY;
				}
				startX += stepStartX;
				startY += stepStartY;
				endX += stepEndX;
				endY += stepEndY;
			}
		}
	},

	createRectMesh: function(resolution) {
		this.rectMesh = new Float32Array(2 * resolution * resolution);
		var stepx = 1.0 / (resolution - 1);
		var stepy = 1.0 / (resolution - 1);
		var x = 0, y = 0;
		var cnt = 0;
		for (var i = 0; i < resolution; ++i) {
			x = 0;
			for (var j = 0; j < resolution; ++j) {
				this.rectMesh[cnt++] = x;
				this.rectMesh[cnt++] = y;
				x += stepx;
			}
			y += stepy;
		}
	},

	doWarp: function() {
		if (!this.node)
			return;
		this.node.setActive();
		this.layerTree.shaderSet.selectShader('filter', 1);
		var FSIZE = this.glMeshArray.BYTES_PER_ELEMENT;
		var gl = this.node.gl;
		gl.clearColor(1.0, 1.0, 1.0, 0.0);
		gl.clear(gl.COLOR_BUFFER_BIT);
		var buffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.bufferData(gl.ARRAY_BUFFER, this.glMeshArray, gl.STATIC_DRAW);
		var a_vertexCoord = gl.getAttribLocation(gl.program, 'a_vertexCoord');
		gl.vertexAttribPointer(a_vertexCoord, 2, gl.FLOAT, false, 4 * FSIZE, 0);
		gl.enableVertexAttribArray(a_vertexCoord);
		var a_texCoord = gl.getAttribLocation(gl.program, 'a_texCoord');
		gl.vertexAttribPointer(a_texCoord, 2, gl.FLOAT, false, 4 * FSIZE, 2 * FSIZE);
		gl.enableVertexAttribArray(a_texCoord);
		var u_texture = gl.getUniformLocation(gl.program, 'u_texture');
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.uniform1i(u_texture, 0);
		var t = this.resolution - 1;
		gl.drawArrays(gl.TRIANGLES, 0, t * t * 6);
	},

	startWarp: function(resolution = 15) {
		if (resolution % 2 == 0)
			++resolution;
		this.createRectMesh(resolution);
		this.glMeshArray = new Float32Array((resolution - 1) * (resolution - 1) * 24);
		this.resolution = resolution;
	},

	updateWarp: function(boundingBox) {
		var assign = function(self, cnt, v1, v2, v3) {
			self.glMeshArray[cnt++] = self.mesh[v1]; // vertex coord
			self.glMeshArray[cnt++] = self.mesh[v1 + 1];
			self.glMeshArray[cnt++] = self.rectMesh[v1]; // tex coord
			self.glMeshArray[cnt++] = self.rectMesh[v1 + 1];
			self.glMeshArray[cnt++] = self.mesh[v2];
			self.glMeshArray[cnt++] = self.mesh[v2 + 1];
			self.glMeshArray[cnt++] = self.rectMesh[v2];
			self.glMeshArray[cnt++] = self.rectMesh[v2 + 1];
			self.glMeshArray[cnt++] = self.mesh[v3];
			self.glMeshArray[cnt++] = self.mesh[v3 + 1];
			self.glMeshArray[cnt++] = self.rectMesh[v3];
			self.glMeshArray[cnt++] = self.rectMesh[v3 + 1];
			return cnt;
		};
		this.createMesh(boundingBox, this.resolution);
		var cnt = 0;
		var resolution = this.resolution;
		var tot = resolution * resolution - resolution;
		for (var i = 0; i < tot; ++i) {
			if ((i + 1) % resolution == 0)
				continue;
			var v1 = i, v2 = v1 + 1, v3 = v1 + resolution;
			v1 *= 2, v2 *= 2, v3 *= 2;
			cnt = assign(this, cnt, v1, v2, v3);
		}
		tot += resolution;
		for (var i = resolution; i < tot; ++i) {
			if ((i + 1) % resolution == 0)
				continue;
			var v1 = i, v2 = v1 + 1, v3 = v2 - resolution;
			v1 *= 2, v2 *= 2, v3 *= 2;
			cnt = assign(this, cnt, v1, v2, v3);
		}
		this.doWarp();
	},

	endWarp: function() {
		this.mesh = null;
		this.meshResolution = -1;
		if (this.texture)
			this.layerTree.gl.deleteTexture(this.tex);
		this.mesh = null;
		this.rectMesh = null;
		this.glMeshArray = null;
		this.meshResolution = -1;
		this.texture = null;
		this.node.selectedPixel = null;
		this.node = null;
	},
};