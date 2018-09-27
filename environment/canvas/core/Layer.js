var LayerTree = function(canvasManager, width, height, devicePixelRatio = 1) {
	this.canvasHeight = height;
	this.canvasWidth = width;
	this.devicePixelRatio = devicePixelRatio;

	this.gl = canvasManager.gl;
	canvasManager.layerTree = this;
	this.shaderSet = new ShaderSet(this.gl);
	this.gl.clearColor(1.0, 1.0, 1.0, 0.0);
	this.gl.clear(this.gl.COLOR_BUFFER_BIT);
	var ext = this.gl.getExtension('OES_element_index_uint');

	var gl = this.gl;
	this.drawTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, this.drawTexture);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
	this.drawBuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, this.drawBuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.drawTexture, 0);

	this.root = new LayerTreeNode(this);
	this.root.nodeId = 0;
	this.root.type = 'folder';
	this.activeNodeId = -1;

	this.blendingMode = {
		'normal': 0,
		'multiply': 1,
		'linear dodge': 2,
		'reflect': 3,
		'linear burn': 4,
		'linear light': 5
	};

	// private cache
	this._searchResult = null;
}

LayerTree.prototype = {
	_searchNode: function(nodeId, root) {
		var p = root;
		if (p.nodeId == nodeId)
			this._searchResult = p;
		else
			for (var i = 0; i < p.children.length; ++i)
				this._searchNode(nodeId, p.children[i]);
	},

	getRootId: function() {
		return this.root.nodeId;
	},

	searchNodeById: function(nodeId, root = this.root) {
		this._searchResult = null;
		this._searchNode(nodeId, root);
		return this._searchResult;
	},

	selectNode: function(nodeId) {
		var node = this.searchNodeById(nodeId);
		if (node != null)
			node.setActive();
		return node;
	},

	positionById: function(nodeId) {
		var node = this.searchNodeById(nodeId);
		var parent = node.parent;
		if (parent == null)
			return 0;	// root id = 0
		for (var i = 0; i < parent.children.length; ++i)
			if (parent.children[i].nodeId == nodeId)
				return i;
		return -1;
	},

	addExistingNode: function(positionInLayer, parentId, node) {
		var parent = this.searchNodeById(parentId);
		var id = new Date().getTime() + Math.round(Math.random() * 10);
		while (this.searchNodeById(id) != null)
			id = new Date().getTime() + Math.round(Math.random() * 10);
		node.nodeId = id;
		node.rendered = false;
		node.parent = parent;
		parent.children.splice(positionInLayer, 0, node);
		return node.nodeId;
	},

	createNewNode: function(positionInLayer, parentId, width = this.canvasWidth, height = this.canvasHeight,
							name = '', type = 'layer') {
		var newNode = new LayerTreeNode(this, width, height);
		newNode.name = name;
		newNode.type = type;
		return this.addExistingNode(positionInLayer, parentId, newNode);
	},

	removeNode: function(nodeId, removeChildren = false) {
		var pos = this.positionById(nodeId);
		var node = this._searchResult;
		var parent = node.parent;
		parent.children.splice(pos, 1);
		if (!removeChildren) {
			for (var i = 0; i < node.children.length; ++i) {
				node.children[i].parent = node.parent;
				parent.children.splice(pos + i, 0, node.children[i]);
			}
			node.destroy(false);
		}
		else
			node.destroy(true);
	},

	moveNode: function(nodeId, destParentId, destPosInParent) {
		var pos = this.positionById(nodeId);
		var node = this.searchNodeById(nodeId);
		var parent = node.parent;
		parent.children.splice(pos, 1);
		var destParent = this.searchNodeById(destParentId);
		destParent.children.splice(destPosInParent, 0, node);
		node.parent = destParent;
		parent.setRenderPath(false);
		destParent.setRenderPath(false);
	},

	copyFromTexture: function(src, verticesTexCoords = null) {
		var gl = this.gl;
		this.shaderSet.selectShader('filter', 0);
		if (verticesTexCoords == null)
			verticesTexCoords = new Float32Array([
				// Vertex coord, tex coord
				-1.0, 1.0, 0.0, 1.0,
				-1.0, -1.0, 0.0, 0.0,
				1.0, 1.0, 1.0, 1.0,
				1.0, -1.0, 1.0, 0.0
			]);

		var n = 4;
		var u_Sampler = gl.getUniformLocation(gl.program, 'u_Sampler');
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, src);
		this.root.setTextureParameter();
		gl.uniform1i(u_Sampler, 0);

		var vertexTexCoordBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexTexCoordBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, verticesTexCoords, gl.STATIC_DRAW);
		
		var FSIZE = verticesTexCoords.BYTES_PER_ELEMENT;
		var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
		gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, FSIZE * 4, 0);
		gl.enableVertexAttribArray(a_Position);

		var a_TexCoord = gl.getAttribLocation(gl.program, 'a_TexCoord');
		gl.vertexAttribPointer(a_TexCoord, 2, gl.FLOAT, false, FSIZE * 4, FSIZE * 2);
		gl.enableVertexAttribArray(a_TexCoord);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
	},

	composite: function(verticesTexCoords = null) {
		this.root.rendered = false;
		this.root.composite(verticesTexCoords);
	},
};

var LayerTreeNode = function(layerTree, width = layerTree.canvasWidth, height = layerTree.canvasHeight) {
	this.children = new Array();
	this.parent = null;
	this.nodeId = -1;
	this.rendered = false;
	this.visible = true;
	this.name = '';
	// subsidiary data for future use, including texture[2]
	this.selectedPixel = null;
	this.width = width;
	this.height = height;
	this.rotation = 0.0;
	this.opacity = 1.0;
	this.scale = 1.0;
	this.blendMode = 0;
	this.areaLeft = 0;
	this.areaBottom = 0;
	this.areaWidth = 0;
	this.areaHeight = 0;
	this.extraTexture = 2;

	this.layerTree = layerTree;
	this.gl = layerTree.gl;
	this.framebuffer = [];
	this.texture = [];
	this.primaryTexture = 0;
	this.secondaryTexture = 1;	// dual buffer

	var gl = this.gl;
	var tex = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, tex);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
	var fbo = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
	this.texture.push(tex, this.layerTree.drawTexture);
	this.framebuffer.push(fbo, this.layerTree.drawBuffer);
	this.texture.push(null);
	this.framebuffer.push(null);
}

LayerTreeNode.prototype = {
	destroy: function(destroyChildren = true) {
		var gl = this.gl;
		gl.deleteTexture(this.texture[0]);
		gl.deleteFramebuffer(this.framebuffer[0]);
		this.selectedPixel = null;
		if (this.texture[2] != null) {
			gl.deleteTexture(this.texture[2]);
			gl.deleteFramebuffer(this.framebuffer[2]);
		}
		this.framebuffer = [], this.texture = [], this.selectedPixel = null;
		if (destroyChildren) {
			for (var i = 0; i < this.children.length; ++i)
				this.children[i].destroy(true);
			this.children = [];
		}
	},

	removeFromParent: function() {
		var parent = this.parent;
		var i;
		if (parent) {
			for (i = 0; i < parent.children.length; ++i) {
				if (parent.children[i].nodeId == this.nodeId) {
					parent.children.splice(i, 1);
					break;
				}
			}
		}
		if (this.layerTree.canvasManager.undoManager) {
			var um = this.layerTree.canvasManager.undoManager;
			um.addStep('delete', this, parent.nodeId, i);
		}
		this.parent = null;
		return this;
	},

	setRenderPath: function(value = false) {
		var p = this;
		while (p.parent != null) {
			p.rendered = value;
			p = p.parent;
		}
		p.rendered = false;
	},

	setVisible: function(f = true) {
		this.visible = f;
		this.setRenderPath();
	},

	setFramebuffer: function(n, node = this) {
		var gl = node.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, node.framebuffer[n]);
		gl.viewport(0, 0, node.width, node.height);
		node.primaryTexture = n;
		node.secondaryTexture = (n == 1 ? 0 : 1);
	},

	setActive: function() {
		this.setFramebuffer(0);
		this.layerTree.activeNodeId = this.nodeId;
	},

	getColor: function(x, y) {
		var clr = new Uint8Array(4);
		var gl = this.gl;
		var id = this.layerTree.activeNodeId;
		this.setActive();
		x = (x + 1.0) * this.width / 2;
		y = (y + 1.0) * this.height / 2;
		gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, clr);
		this.layerTree.searchNodeById(id).setActive();
		return clr;
	},

	getLayerImageData: function(area = null) {
		var id = this.layerTree.activeNodeId;
		this.setActive();
		var gl = this.gl;
		var pixels = new Uint8Array(this.width * this.height * 4);
		if (area == null)
			gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
		else {
			// TODO: get the rect and readPixels
		}
		return pixels;
	},

	// deprecated
	setLayerImage: function(image) {
		var gl = this.gl;
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this.texture[0]);
		this.setTextureParameter();
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
	},

	setLayerImageData: function(data, x = 0, y = 0, width = this.width, height = this.height) {
		var gl = this.gl;
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this.texture[0]);
		this.setTextureParameter(false);
		gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);
	},

	setTextureParameter: function(premultiplyAlpha = true, flipY = true) {
		var gl = this.gl;
		gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premultiplyAlpha);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flipY);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	},

	nodeCoordTransform: function(coord) {
		var result = new Array();
		var cnt = 0;
		var t = new CoordTransform();
		var n = Math.floor(coord.length / 2);
		for (var i = 0; i < n; ++i) {
			t.setParameter(coord[cnt], coord[cnt + 1], this.rotation, this.scale, this.offsetX, this.offsetY);
			cnt += 2;
			result.push(t.yieldX(), t.yieldY());
		}
		return result;
	},

	alphaBlend: function(verticesTexCoords, baseTexture, newTexture, baseOpacity, newOpacity) {
		var n = 4;
		var gl = this.gl;
		var vertexTexCoordBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexTexCoordBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, verticesTexCoords, gl.STATIC_DRAW);

		var FSIZE = verticesTexCoords.BYTES_PER_ELEMENT;
		var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
		gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, FSIZE * 4, 0);
		gl.enableVertexAttribArray(a_Position);

		var a_TexCoord = gl.getAttribLocation(gl.program, 'a_TexCoord');
		gl.vertexAttribPointer(a_TexCoord, 2, gl.FLOAT, false, FSIZE * 4, FSIZE * 2);
		gl.enableVertexAttribArray(a_TexCoord);

		var u_ChildOpacity = gl.getUniformLocation(gl.program, 'u_ChildOpacity');
		gl.uniform1f(u_ChildOpacity, newOpacity);
		var u_CanvasOpacity = gl.getUniformLocation(gl.program, 'u_CanvasOpacity');
		gl.uniform1f(u_CanvasOpacity, baseOpacity);

		var u_CanvasSampler = gl.getUniformLocation(gl.program, 'u_CanvasSampler');
		var u_ChildSampler = gl.getUniformLocation(gl.program, 'u_ChildSampler');

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, baseTexture);
		this.setTextureParameter();
		gl.uniform1i(u_CanvasSampler, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, newTexture);
		this.setTextureParameter();
		gl.uniform1i(u_ChildSampler, 1);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
	},

	multiplyBlend: function(verticesTexCoords, baseTexture, newTexture, baseOpacity, newOpacity) {
		this.alphaBlend(verticesTexCoords, baseTexture, newTexture, baseOpacity, newOpacity);
	},

	composite: function(verticesTexCoords = null) {
		var gl = this.gl;
		for (var i = 0; i < this.children.length; ++i) {
			this.children[i].composite(verticesTexCoords);
		}

		if (verticesTexCoords == null)
			verticesTexCoords = new Float32Array([
				// vertex coord  tex coord
				-1.0, 1.0,       0.0, 1.0,
				-1.0, -1.0,      0.0, 0.0,
				1.0, 1.0,        1.0, 1.0,
				1.0, -1.0,       1.0, 0.0
			]);
		var n = 4;
		if (this.children.length && !this.rendered && this.type == 'folder') {
			var t = 0;
			for (var i = 0; i < this.children.length; ++i)
				if (this.children[i].visible)
					++t;
			// debug the composition algorithm
			// clear tmp buffer
			this.setFramebuffer(1);
			gl.clearColor(1.0, 1.0, 1.0, 0.0);
			gl.clear(gl.COLOR_BUFFER_BIT);
			// clear rect
			this.setFramebuffer(0);
			gl.enable(gl.SCISSOR_TEST);
			var x, y, w, h;
			x = Math.floor((1.0 + verticesTexCoords[4]) * this.width / 2) + 1;
			y = Math.floor((1.0 + verticesTexCoords[5]) * this.height / 2) + 1;
			w = (verticesTexCoords[8] - verticesTexCoords[0]) * this.width / 2 - 1;
			h = (verticesTexCoords[9] - verticesTexCoords[5]) * this.height / 2 - 1;
			gl.scissor(x, y, w, h);
			gl.clear(gl.COLOR_BUFFER_BIT);
			gl.disable(gl.SCISSOR_TEST);
			for (var i = this.children.length - 1; i >= 0; --i) {
				if (this.children[i].visible) {
					var child = this.children[i];
					this.setFramebuffer(--t & 1);
					this.layerTree.shaderSet.selectShader('blend', child.blendMode);
					switch (child.blendMode) {
					case 0:
						this.alphaBlend(verticesTexCoords, this.texture[this.secondaryTexture],
										child.texture[0], this.opacity, child.opacity);
						break;
					case 1: 
					case 2:
						this.multiplyBlend(verticesTexCoords, this.texture[this.secondaryTexture],
										child.texture[0], this.opacity, child.opacity);
						break;
					default:
						break;
					}
				}
			}
		}
		this.rendered = true;
	},

	merge: function() {
		this.rendered = false;
		this.composite();
		var children = this.children;
		this.children = new Array();
		if (this.layerTree.canvasManager.undoManager) {
			var um = this.layerTree.canvasManager.undoManager;
			um.addStep('merge', this, children, this.layerTree.positionById(this.nodeId));
		}
		this.type = 'layer';
	},

	is_edge_table_empty: function(edge_table) {
		for (var i = 0; i < edge_table.length; ++i) {
			if (edge_table[i].length != 0)
				return false;
		}
		return true;
	},

	// use original coordinate
	selectArea: function(boundary, substract = false) {
		var gl = this.gl;
		this.setActive();
		var left = 65536, right = -65536, top = -65536, bottom = 65536;
		var left_i, right_i, top_i, bottom_i;
		for (var i = 0; i < boundary.length; i += 2) {
			boundary[i + 1] = this.height - boundary[i + 1];
			x = boundary[i], y = boundary[i + 1];
			if (x < left)
				left = x, left_i = i;
			if (x > right)
				right = x, right_i = i;
			if (y > top)
				top = y, top_i = i;
			if (y < bottom)
				bottom = y, bottom_i = i;
		}
		var rectHeight = top - bottom;
		var rectWidth = right - left;
		var len = 4 * rectWidth * rectHeight;
		var pixelBuffer = new Uint8Array(len);
		var destBuffer = new Uint8Array(pixelBuffer.length);
		for (var i = 0; i < len; ++i) {
			destBuffer[i++] = 255;
			destBuffer[i++] = 255;
			destBuffer[i++] = 255;
			destBuffer[i] = 0;
		}
		boundary.push(boundary[0], boundary[1]);
		gl.readPixels(left, bottom, rectWidth, rectHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);

		var len = boundary.length / 2;
		var tmp = top - bottom + 1;
		var edge_table = new Array(tmp);
		for (var i = 0; i < tmp; ++i)
			edge_table[i] = new Array();
		var active_edge_table = new Array();
		var pt_num = len - 1;
		var y = 65536;

		tmp = pt_num * 2;
		for (var i = 0; i < tmp; i += 2) {
			var y_upper = boundary[i + 1] > boundary[i + 3] ? boundary[i + 1] : boundary[i + 3];
			var y_lower = boundary[i + 1] < boundary[i + 3] ? boundary[i + 1] : boundary[i + 3];
			var x_lower = boundary[i + 1] < boundary[i + 3] ? boundary[i] : boundary[i + 2];
			var dx = boundary[i] - boundary[i + 2];
			var dy = boundary[i + 1] - boundary[i + 3];
			var k;
			if (dy != 0)
				k = dx / dy;
			else continue;
			edge_table[y_lower - bottom].push({x: x_lower, dx: k, y_upper: y_upper});
			if (boundary[i + 1] < y)
	            y = boundary[i + 1];
		}
		var _y = y - bottom;
		while (!this.is_edge_table_empty(edge_table) || active_edge_table.length != 0) {
			var idx = y - bottom;
			if (edge_table[idx].length != 0) {
				for (var i = 0; i < edge_table[idx].length; ++i)
					active_edge_table.push({x: edge_table[idx][i].x, dx: edge_table[idx][i].dx, y_upper: edge_table[idx][i].y_upper});
				edge_table[idx] = [];
			}
			active_edge_table.sort(function(a, b) {
				if (a.x != b.x)
					return a.x - b.x;
				return a.dx - b.dx;
			});
			for (var i = 0; i < active_edge_table.length; ++i) {
				var x0 = Math.floor(active_edge_table[i].x), x1 = Math.floor(active_edge_table[++i].x);
				var _x = x0 - left;
				var t = 4 * (_y * rectWidth + _x);
				for (var x = x0; x <= x1; ++x) {
					destBuffer[t + 0] = pixelBuffer[t + 0];
					destBuffer[t + 1] = pixelBuffer[t + 1];
					destBuffer[t + 2] = pixelBuffer[t + 2];
					destBuffer[t + 3] = pixelBuffer[t + 3];
					if (substract) {
						pixelBuffer[t + 0] = 0;
						pixelBuffer[t + 1] = 0;
						pixelBuffer[t + 2] = 0;
						pixelBuffer[t + 3] = 0;
					}
					++_x;
					t += 4;
				}
			}
			++y;
			++_y;
			for (var i = 0; i < active_edge_table.length; ) {
				if (active_edge_table[i].y_upper == y)
					active_edge_table.splice(i, 1);
				else {
					active_edge_table[i].x += active_edge_table[i].dx;
					++i;
				}
			}
		}
		var ret = new LayerTreeNode(this.layerTree, this.width, this.height);
		ret.opacity = this.opacity;
		if (ret.selectedPixel != null)
			delete ret.selectedPixel;
		ret.selectedPixel = destBuffer;
		ret.type = 'layer';
		var areaTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, areaTexture);
		this.setTextureParameter(false, false);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, rectWidth, rectHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, destBuffer);
		gl.bindTexture(gl.TEXTURE_2D, ret.texture[ret.primaryTexture]);
		ret.areaWidth = rectWidth;
		ret.areaHeight = rectHeight;
		ret.areaLeft = left;
		ret.areaBottom = bottom;

		var coord = new CoordTransform();
		coord.setCanvasMetrics(this.width, this.height, this.layerTree.devicePixelRatio);
		var t = new Array(8);
		coord.canvasCoord2GL(left, this.height - top);
		t[0] = coord.x, t[1] = coord.y;
		coord.canvasCoord2GL(left, this.height - bottom);
		t[2] = coord.x, t[3] = coord.y;
		coord.canvasCoord2GL(right, this.height - top);
		t[4] = coord.x, t[5] = coord.y;
		coord.canvasCoord2GL(right, this.height - bottom);
		t[6] = coord.x, t[7] = coord.y;

		var verticesTexCoords = new Float32Array([
			// vertex coord, tex coord
			t[0], t[1],      0.0, 1.0,
			t[2], t[3],      0.0, 0.0,
			t[4], t[5],      1.0, 1.0,
			t[6], t[7],      1.0, 0.0
		]);
		ret.setActive();
		this.layerTree.copyFromTexture(areaTexture, verticesTexCoords);
		this.setActive();
		if (substract) {
			gl.bindTexture(gl.TEXTURE_2D, areaTexture);
			this.setTextureParameter(false, false);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, rectWidth, rectHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);
			this.layerTree.copyFromTexture(areaTexture, verticesTexCoords);
			this.setRenderPath(false);
		}
		gl.deleteTexture(areaTexture);
		if (this.layerTree.canvasManager.undoManager) {
			var um = this.layerTree.canvasManager.undoManager;
			um.addStep('area', ret, new Float32Array([-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0,
						1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0]));
		}
		return ret;
	}
};