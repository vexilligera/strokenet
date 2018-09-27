function clamp(val, min, max) {
	if (val < min)
		return min;
	else if (val > max)
		return max;
	return val;
}

function rand(min = 0.0, max = 1.0, int = false, seed = 0) {
	max = max;
	min = min;
	seed = (seed * 9301 + 49297) % 233280;
	var rnd = seed / 233280.0;
	var ret = min + rnd * (max - min);
	if (int)
		return Math.round(ret);
	return ret;
}

var Brush = function(id) {
	this.brushId = id;
	this.name = '';
	this.opacity = 1.0;
	this.density = 1.0;
	this.polygon = 256;
	this.pressureSizeSensitivity = 1.0;
	this.pressureColorSensitivity = 1.0;
	this.hotkey = '';
	this.innerThreshold = 0.5;
	this.blendingMode = 'NORMAL';
	this.fixedColor = [-1, -1, -1, -1];
	// color mixing
	this.enableColorMixing = false;
	this.mixThreshold = 0.5;
	this.mixStrength = 0.5;
	// jittering
	this.enableJitter = false;
	this.sizeJitter = 0.0;
	this.positionJitter = 0.0;
	// tilting enabled
	this.tiltShading = false;
	this.tiltSensitivity = 1.0;
	// texture path, load from file
	this.texture = '';
	// filter parameters
	this.filterType = '';
	this.kernelSize = 8;
	this.sigma = 1.0;
};

Brush.prototype = {
	loadSetting: function(settings) {
		this.name = settings.name;
		this.opacity = clamp(settings.opacity, 0.0, 1.0);
		this.density = clamp(settings.density, 0.0, 5.0);
		this.polygon = clamp(settings.polygon, 3, 512);
		this.pressureSizeSensitivity = clamp(settings.pressureSizeSensitivity, 0.0, 5.0);
		this.pressureColorSensitivity = clamp(settings.pressureColorSensitivity, 0.0, 5.0);
		this.hotkey = settings.hotkey.toUpperCase();
		this.innerThreshold = clamp(settings.innerThreshold, 0.0, 1.0);
		this.blendingMode = settings.blendingMode;
		if (settings.fixedColor != undefined && settings.fixedColor[0] >= 0)
			this.fixedColor[0] = settings.fixedColor[0], this.fixedColor[1] = settings.fixedColor[1],
			this.fixedColor[2] = settings.fixedColor[2], this.fixedColor[3] = settings.fixedColor[3];
		this.enableColorMixing = settings.enableColorMixing;
		this.mixThreshold = clamp(settings.mixThreshold, 0.0, 1.0);
		this.mixStrength = clamp(settings.mixStrength, 0.0, 1.0);
		this.enableJitter = settings.enableJitter;
		this.sizeJitter = clamp(settings.sizeJitter, 0.0, 1.0);
		this.positionJitter = clamp(settings.positionJitter, 0.0, 5.0);
		this.tiltShading = settings.tiltShading;
		this.tiltSensitivity = settings.tiltSensitivity;
		this.texture = settings.texture.toString();
		if (settings.sigma != undefined)
			this.sigma = settings.sigma;
		if (settings.filterType != undefined)
			this.filterType = settings.filterType;
		if (settings.kernelSize != undefined)
			this.kernelSize = settings.kernelSize;
		return this.brushId;
	}
};

var BrushSet = function(layerTree) {
	this.count = 0;
	this.brushes = new Array();
	this.activeBrushIndex = -1;
	this.layerTree = layerTree;
	this.gl = layerTree.gl;
	this.activeNode = null;
	this.shaderSet = layerTree.shaderSet;
	this.drawTexture = layerTree.drawTexture;
	this.drawBuffer = layerTree.drawBuffer;
	this.undoManager = null;
	this.strokeRect = null;

	this.radius = 0;
	this.brushTexture = this.gl.createTexture();
	this.brushTextureUnit = 5;

	var factor = layerTree.canvasWidth > layerTree.canvasHeight ? layerTree.canvasWidth / layerTree.canvasHeight : layerTree.canvasHeight / layerTree.canvasWidth;
	this.horizontalFactor = layerTree.canvasWidth > layerTree.canvasHeight ? 1.0 : factor;
	this.verticalFactor = layerTree.canvasWidth < layerTree.canvasHeight ? 1.0 : factor;

	this.color0 = [0.0, 0.0, 0.0, 1.0];	// normalized RGBA
	this.color1 = [1.0, 1.0, 1.0, 1.0];
	// stroke stablizer
	var gl = this.gl;
	/*
	this.samplerTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, this.samplerTexture);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, layerTree.canvasWidth, layerTree.canvasHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
	this.samplerBuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, this.samplerBuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.samplerTexture, 0);
	*/
	this.strokeCache = new Array(11);
	this.interpolateCache = new Array(11);
	this.interpolateLength = 0;
	for (var i = 0; i < 11; ++i) {
		this.strokeCache[i] = new Array(4);
		this.interpolateCache[i] = new Float32Array(256);
	}
	this.onStroke = false;
	this.strokeCacheIndex = -1;
	this.curve = new CatmullRomSpline(256);
	this.brushComposite = new Array(4);

	// memory
	this._glInputArray = new Float32Array(65536 * 4);
	this._indices = new Uint32Array(5120 * 4);
	this._invalidRect = new Float32Array(16);
}

BrushSet.prototype = {
	selectBrush: function(brushId, radius = 0.05) {
		this.activeBrushIndex = brushId;
		this.radius = radius;
		// load texture if have
		var gl = this.gl;
		var brush = this.brushes[this.activeBrushIndex];
		if (brush.texture != '') {
			var image = new Image();
			image.crossOrigin = 'anonymous';
			var _self = this;
			image.onload = function() {
				gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
				gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
				gl.activeTexture(gl.TEXTURE5);	// brush texture in unit 5
				gl.bindTexture(gl.TEXTURE_2D, _self.brushTexture);
				gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
			};
			image.src = brush.texture;
		}
		/*
		else {
			gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
		}
		*/
		if (brush.blendingMode == 'NORMAL' || brush.blendingMode == 'ERASER') {
			this.brushComposite[0] = gl.ONE;
			this.brushComposite[1] = gl.ZERO;
			this.brushComposite[2] = gl.ONE;
			this.brushComposite[3] = gl.ONE_MINUS_SRC_ALPHA;
		}
		else if (brush.blendingMode == 'FILTER') {
			this.kernelSize = brush.kernelSize;
			this.sigma = brush.sigma;
			this.filterType = brush.filterType;
			this.kernel = null;
			if (this.filterType == 'gaussian')
				this.kernel = genGaussianKernel(this.kernelSize, this.sigma);
		}
		else if (brush.blendingMode == 'DIRECT') {
			this.brushComposite[0] = gl.SRC_ALPHA;
			this.brushComposite[1] = gl.ONE_MINUS_SRC_ALPHA;
			this.brushComposite[2] = gl.ONE;
			this.brushComposite[3] = gl.ONE_MINUS_SRC_ALPHA;
		}
	},

	setRadius: function(radius) {
		this.radius = radius;
		this._frac_radius = radius / 10;
	},

	setPrimaryColor: function(color) {
		for (var i = 0; i < 4; ++i)
			this.color0[i] = color[i];
	},

	setSecondaryColor: function(color) {
		for (var i = 0; i < 4; ++i)
			this.color1[i] = color[i];
	},

	loadBrushes: function(brushArray, reload = false) {
		if (reload)
			this.count = 0;
		for (var i = 0; i < brushArray.length; ++i) {
			this.brushes[this.count] = new Brush(this.count);
			this.brushes[this.count].loadSetting(brushArray[i]);
			++this.count;
		}
		return this.count;
	},

	beginDraw: function(id) {
		this.activeNode = this.layerTree.searchNodeById(id);
		this.coordStepX = 2.0 / this.activeNode.width;
		this.coordStepY = 2.0 / this.activeNode.height;
	},

	endDraw: function() {
		if (this.activeNode != null) {
			this.activeNode = null;
		}
	},

	bindUndoManager: function(um) {
		this.undoManager = um;
	},

	beginStroke: function(id) {
		this.beginDraw(id);
		this.onStroke = true;
		this.strokeCacheIndex = 0;
		this.diameter = this.radius * 2;
		this.prevRadius = -1;
		// setup
		var gl = this.gl;
		var layerTree = this.layerTree;
		
		var brush = this.brushes[this.activeBrushIndex];
		if (brush.fixedColor[0] >= 0) {
			this.paletteCache = this.color0;
			this.color0 = this.fixedColor;
		}
		// TODO: use a dedicated buffer node to optimize
		if (brush.blendingMode == 'NORMAL' || brush.blendingMode == 'FILTER' || 
			brush.blendingMode == 'ERASER') {
			this.brushBuffer = layerTree.createNewNode(0, layerTree.getRootId());
			var pos = layerTree.positionById(this.activeNode.nodeId);
			if (pos > 0 && this.activeNode.parent.children[pos - 1].blendMode != 1)
				--pos;
			pos = pos < 0 ? 0 : pos;
			layerTree.moveNode(this.brushBuffer, this.activeNode.parent.nodeId, pos);
			this.brushBufferNode = layerTree.searchNodeById(this.brushBuffer);
			this.brushBufferNode.name = 'BRUSH_NORMAL_BUFFER';
			this.brushBufferNode.opacity = this.activeNode.opacity;
			if (brush.blendingMode == 'ERASER') {
				this.brushBufferNode.setActive();
				gl.clearColor(1.0, 1.0, 1.0, 0.0);
				gl.clear(gl.COLOR_BUFFER_BIT);
				this.brushBufferNode.blendMode = 2; // stencil
			}
		}
		else
			this.brushBufferNode = this.activeNode;
		this.brushBufferNode.setActive();
	},

	endStroke: function() {
		this.strokeTo(symmetric(this.strokeCache[0][2], this.strokeCache[0][1]), symmetric(this.strokeCache[1][2], this.strokeCache[1][1]),
						symmetric(this.strokeCache[2][2], this.strokeCache[2][1]), 0, 0, [this.strokeCache[5][2], this.strokeCache[6][2], 
						this.strokeCache[7][2], this.strokeCache[8][2]], symmetric(this.strokeCache[9][2], this.strokeCache[9][1]),
						this.strokeCache[10][2]);
		this.prevRadius = -1;
		var gl = this.gl;
		var brush = this.brushes[this.activeBrushIndex];
		if (brush.blendingMode == 'NORMAL' || brush.blendingMode == 'FILTER' || brush.blendingMode == 'ERASER') {
			var coord = new Float32Array([-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0]);
			if (this.undoManager) {
				this.undoManager.addStep('area', this.activeNode, this.strokeRect);
				this.strokeRect = null;
			}
			gl.bindFramebuffer(gl.FRAMEBUFFER, this.drawBuffer);
			gl.viewport(0, 0, this.activeNode.width, this.activeNode.height);
			this.layerTree.shaderSet.selectShader('blend', this.brushBufferNode.blendMode);
			if (this.brushBufferNode.blendMode == 0 || this.brushBufferNode.blendMode == 1)
				this.activeNode.alphaBlend(coord, this.activeNode.texture[this.activeNode.primaryTexture], this.brushBufferNode.texture[0], 1.0, 1.0);
			else if (this.brushBufferNode.blendMode == 2)
				this.activeNode.multiplyBlend(coord, this.activeNode.texture[this.activeNode.primaryTexture], this.brushBufferNode.texture[0], 1.0, 1.0);
			this.activeNode.setActive();
			this.layerTree.copyFromTexture(this.drawTexture);
			this.layerTree.removeNode(this.brushBuffer, true);
		}
		if (brush.fixedColor[0] >= 0)
			this.color0 = this.paletteCache;
		this.onStroke = false;
		this.endDraw();
	},

	setStrokeCache: function(index, x, y, pressure, tiltX, tiltY, color, time, radius) {
		this.strokeCache[0][index] = x;
		this.strokeCache[1][index] = y;
		this.strokeCache[2][index] = pressure;
		this.strokeCache[3][index] = tiltX;		// normalized tilting angle -1.0 to 1.0
		this.strokeCache[4][index] = tiltY;
		this.strokeCache[5][index] = color[0];	// normalized RGBA 0.0 to 1.0
		this.strokeCache[6][index] = color[1];
		this.strokeCache[7][index] = color[2];
		this.strokeCache[8][index] = color[3];
		this.strokeCache[9][index] = time;
		this.strokeCache[10][index] = radius;
	},

	drawStroke: function(interpolateCache, invalidRect) {
		var n0, totalCount = 0, offset = 13, idx = 0, vertexCount = 0;
		var brush = this.brushes[this.activeBrushIndex];
		var _nArray = new Array();
		// generate vertices first
		for (var i = 0; i < this.interpolateLength; ++i) {
			n0 = brush.polygon * Math.sqrt(interpolateCache[10][i]);
			_nArray.push(n0 + 3);
			totalCount += n0 + 3;
		}

		var glInputArray = this._glInputArray;
		var indices = this._indices;
		var cnt = 0, icnt = 0;
		for (var i = 0; i < this.interpolateLength; ++i) {
			var _n = _nArray[i];
			var theta = 0.0;
			var step = Math.PI * 2 / (_n - 3);

			for (var j = 0; j < offset; ++j)
				glInputArray[cnt++] = interpolateCache[j % 11][i];
			indices[icnt++] = idx++;
			++vertexCount;
			var _i;
			for (var j = 3; j < _n; ++j) {
				_i = cnt;
				for (var k = 0; k < offset; ++k)
					glInputArray[cnt++] = interpolateCache[k % 11][i];

				// polygon brush with tilting, needs a factor here
				var _tmpx, _tmpy;
				if (brush.tiltSensitivity > 0 && brush.tiltShading) {
					_tmpx = brush.tiltSensitivity * interpolateCache[10][i] * (1.0 + interpolateCache[3][i]);
					_tmpy = brush.tiltSensitivity * interpolateCache[10][i] * (1.0 + interpolateCache[4][i]);
				}
				else
					_tmpx = 0.0, _tmpy = 0.0;

				glInputArray[_i++] += (Math.cos(theta) + interpolateCache[3][i] * _tmpx) * this.horizontalFactor * this.interpolateCache[10][i];
				glInputArray[_i++] += (Math.sin(theta) + interpolateCache[4][i] * _tmpy) * this.verticalFactor * this.interpolateCache[10][i];

				indices[icnt++] = idx++;
				theta += step;
				++vertexCount;
			}
			for (var k = 0; k < offset; ++k)
				glInputArray[cnt++] = interpolateCache[k % 11][i];
			glInputArray[cnt - 13] += Math.cos(0) * interpolateCache[10][i] * this.horizontalFactor;
			glInputArray[cnt - 12] += Math.sin(0) * interpolateCache[10][i] * this.verticalFactor;

			indices[icnt++] = idx++;
			theta += step;
			++vertexCount;

			indices[icnt++] = 0xffffffff;
			vertexCount++;
		}
		var gl = this.gl;
		var layerTree = this.layerTree;
		this.brushBufferNode.setActive();
		switch (brush.blendingMode) {
		case 'ERASER':
		case 'NORMAL':
			this.shaderSet.selectShader('brush', 0);	// general brush shader
			break;
		case 'FILTER':
			this.shaderSet.selectShader('filter', 2);	// gaussian blur shader
			break;
		}

		var indexBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW, 0, totalCount + this.interpolateLength);

		var vertexBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, glInputArray, gl.STATIC_DRAW, 0, totalCount * offset);
		var FSIZE = glInputArray.BYTES_PER_ELEMENT;
		var stride = offset * FSIZE;

		if (brush.blendingMode == 'NORMAL' || brush.blendingMode == 'ERASER') {
			var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
			gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, stride, 0);
			gl.enableVertexAttribArray(a_Position);

			var a_Pressure = gl.getAttribLocation(gl.program, 'a_Pressure');
			gl.vertexAttribPointer(a_Pressure, 1, gl.FLOAT, false, stride, 2 * FSIZE);
			gl.enableVertexAttribArray(a_Pressure);

			var a_Tilt = gl.getAttribLocation(gl.program, 'a_Tilt');
			gl.vertexAttribPointer(a_Tilt, 2, gl.FLOAT, false, stride, 3 * FSIZE);
			gl.enableVertexAttribArray(a_Tilt);

			var a_Color = gl.getAttribLocation(gl.program, 'a_Color');
			gl.vertexAttribPointer(a_Color, 4, gl.FLOAT, false, stride, 5 * FSIZE);
			gl.enableVertexAttribArray(a_Color);

			var a_Time = gl.getAttribLocation(gl.program, 'a_Time');
			gl.vertexAttribPointer(a_Time, 1, gl.FLOAT, false, stride, 9 * FSIZE);
			gl.enableVertexAttribArray(a_Time);

			var a_Radius = gl.getAttribLocation(gl.program, 'a_Radius');
			gl.vertexAttribPointer(a_Radius, 1, gl.FLOAT, false, stride, 10 * FSIZE);
			gl.enableVertexAttribArray(a_Radius);

			var a_CenterPos = gl.getAttribLocation(gl.program, 'a_CenterPos');
			gl.vertexAttribPointer(a_CenterPos, 2, gl.FLOAT, false, stride, 11 * FSIZE);
			gl.enableVertexAttribArray(a_CenterPos);

			var u_layerSampler = gl.getUniformLocation(gl.program, 'u_layerSampler');
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, this.activeNode.texture[0]);
			gl.uniform1i(u_layerSampler, 0);

			var u_xyFactor = gl.getUniformLocation(gl.program, 'u_xyFactor');
			gl.uniform2fv(u_xyFactor, [this.horizontalFactor, this.verticalFactor]);

			var u_InnerThreshold = gl.getUniformLocation(gl.program, 'u_InnerThreshold');
			gl.uniform1f(u_InnerThreshold, brush.innerThreshold);
			// load brush texture
			var u_brushHasTexture = gl.getUniformLocation(gl.program, 'u_brushHasTexture');
			var u_brushTexture = gl.getUniformLocation(gl.program, 'u_brushTexture');
			if (brush.texture != '') {	// if has texture
				gl.activeTexture(gl.TEXTURE5);
				gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
				gl.uniform1i(u_brushTexture, this.brushTextureUnit);	// unit 5
				gl.uniform1i(u_brushHasTexture, 1);
			}
			else {
				gl.uniform1i(u_brushHasTexture, 0);
				gl.uniform1i(u_brushTexture, 0);
			}
			// tilt option
			var u_tiltSensitivity = gl.getUniformLocation(gl.program, 'u_tiltSensitivity');
			var inputsensitivity = brush.tiltShading ? brush.tiltSensitivity  : -1.0;
			gl.uniform1f(u_tiltSensitivity, inputsensitivity);

			// color mixing option
			var inputmixthreshold = brush.enableColorMixing ? brush.mixThreshold : -1.0;
			var u_mixThreshold = gl.getUniformLocation(gl.program, 'u_mixThreshold');
			gl.uniform1f(u_mixThreshold, inputmixthreshold);
			if (brush.enableColorMixing) {
				var u_mixStrength = gl.getUniformLocation(gl.program, 'u_mixStrength');
				gl.uniform1f(u_mixStrength, brush.mixStrength);
			}

			// pressure
			var u_pressureColorSensitivity = gl.getUniformLocation(gl.program, 'u_pressureColorSensitivity');
			gl.uniform1f(u_pressureColorSensitivity, brush.pressureColorSensitivity);

			gl.enable(gl.BLEND);
			gl.blendFuncSeparate(this.brushComposite[0], this.brushComposite[1], this.brushComposite[2], this.brushComposite[3]);
			this.brushBufferNode.setTextureParameter();
			gl.drawElements(gl.TRIANGLE_FAN, vertexCount, gl.UNSIGNED_INT, 0);
			this.brushBufferNode.setRenderPath(false);
			gl.disable(gl.BLEND);
		}
		else if (brush.blendingMode == 'FILTER') {
			var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
			gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, stride, 0);
			gl.enableVertexAttribArray(a_Position);

			var a_Pressure = gl.getAttribLocation(gl.program, 'a_Pressure');
			gl.vertexAttribPointer(a_Pressure, 1, gl.FLOAT, false, stride, 2 * FSIZE);
			gl.enableVertexAttribArray(a_Pressure);

			var u_layerSampler = gl.getUniformLocation(gl.program, 'u_layerSampler');
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, this.activeNode.texture[0]);
			gl.uniform1i(u_layerSampler, 0);

			var u_xyFactor = gl.getUniformLocation(gl.program, 'u_xyStep');
			gl.uniform2fv(u_xyFactor, [this.coordStepX, this.coordStepY]);

			var u_kernelSize = gl.getUniformLocation(gl.program, 'u_kernelSize');
			gl.uniform1f(u_kernelSize, this.kernelSize);

			var u_kernel = gl.getUniformLocation(gl.program, 'u_kernel');
			gl.uniform1fv(u_kernel, this.kernel);

			gl.drawElements(gl.TRIANGLE_FAN, vertexCount, gl.UNSIGNED_INT, 0);
			this.brushBufferNode.setRenderPath(false);
		}
	},
	
	strokeTo: function(x, y, pressure = 1.0, tiltX = 0.0, tiltY = 0.0, color = this.color0, time = 0, radius = this.radius) {
		var brush = this.brushes[this.activeBrushIndex];
		radius -= (1.0 - pressure) * brush.pressureSizeSensitivity * radius;
		if (this.prevRadius < 0) this.prevRadius = radius;
		if (radius / this.prevRadius > 1.1)
			radius = this.prevRadius * 1.1;
		else if (radius / this.prevRadius < 0.9)
			radius = this.prevRadius * 0.9;
		this.prevRadius = radius;
		time = time % 1000000 * 1000;

		if (this.strokeCacheIndex == 0) {
			++this.strokeCacheIndex;
			this.setStrokeCache(this.strokeCacheIndex, x, y, pressure, tiltX, tiltY, color, time, radius);
		}
		else if (this.strokeCacheIndex == 1) {
			++this.strokeCacheIndex;
			this.setStrokeCache(this.strokeCacheIndex, x, y, pressure, tiltX, tiltY, color, time, radius);
			this.setStrokeCache(0, symmetric(this.strokeCache[0][1], x), symmetric(this.strokeCache[1][1], y),
								symmetric(this.strokeCache[2][1], pressure), symmetric(this.strokeCache[3][1], tiltX),
								symmetric(this.strokeCache[4][1], tiltY), [symmetric(this.strokeCache[5][1], color[0]),
								symmetric(this.strokeCache[6][1], color[1]), symmetric(this.strokeCache[7][1], color[2]),
								symmetric(this.strokeCache[8][1], color[3])], symmetric(this.strokeCache[9][1], time),
								symmetric(this.strokeCache[10][1], radius));
		}
		else if (this.strokeCacheIndex == 2) {
			++this.strokeCacheIndex;
			this.setStrokeCache(this.strokeCacheIndex, x, y, pressure, tiltX, tiltY, color, time, radius);

			var dx, dy, dist, n;
			dx = this.strokeCache[0][2] - this.strokeCache[0][1];
			dy = this.strokeCache[1][2] - this.strokeCache[1][1];
			dist = Math.sqrt(dx * dx + dy * dy);
			n = 4 * Math.floor(dist / radius) * brush.density;
			if (n < 4)
				n = 4;
			else if (n > 256)
				n = 256;

			this.interpolateLength = n;
			// catmull-rom interpolate the curve
			if (Math.max(Math.abs(this.strokeCache[0][1] - this.strokeCache[0][2]), Math.abs(this.strokeCache[1][1] - this.strokeCache[1][2])) <= this._frac_radius) {
				linearInterpolate(this.strokeCache[0][1], this.strokeCache[0][2], n, this.interpolateCache[0]);
				linearInterpolate(this.strokeCache[1][1], this.strokeCache[1][2], n, this.interpolateCache[1]);
			}
			else
				this.curve.interpolate(this.strokeCache[0], this.strokeCache[1], n, this.interpolateCache[0], this.interpolateCache[1]);
			// linearly interpolate other data
			for (var i = 2; i < 11; ++i) {
				linearInterpolate(this.strokeCache[i][1], this.strokeCache[i][2], n, this.interpolateCache[i]);
			}
			// Jitter control
			if (brush.enableJitter) {
				var len = this.interpolateCache[0].length;
				// position jitter
				var positionMaxJitter = this.radius * brush.positionJitter;
				for (var i = 0; i < len; ++i) {
					this.interpolateCache[0][i] += rand(0.0, positionMaxJitter, false, this.interpolateCache[9][i]);
					this.interpolateCache[1][i] += rand(0.0, positionMaxJitter, false, this.interpolateCache[9][i]);
				}
				// size jitter
				var sizeJitter = brush.sizeJitter * radius;
				for (var i = 0; i < len; ++i)
					this.interpolateCache[10][i] += rand(-sizeJitter, sizeJitter, false, this.interpolateCache[9][i]);
			}

			var left, right, top, bottom;
			if (this.strokeCache[0][2] < this.strokeCache[0][1])
				left = this.strokeCache[0][2], right = this.strokeCache[0][1];
			else
				right = this.strokeCache[0][2], left = this.strokeCache[0][1];
			if (this.strokeCache[1][2] > this.strokeCache[1][1])
				top = this.strokeCache[1][2], bottom = this.strokeCache[1][1];
			else
				bottom = this.strokeCache[1][2], top = this.strokeCache[1][1];
			if (brush.enableJitter) {
				left -= brush.positionJitter * this.radius, right += brush.positionJitter * this.radius;
				bottom -= brush.positionJitter * this.radius, top += brush.positionJitter * this.radius;
			}
			var t = radius * 2;
			left -= t, right += t, bottom -= t, top += t;
			var _left = (left + 1) / 2, _right = (right + 1) / 2, _top = (top + 1) / 2, _bottom = (bottom + 1) / 2;
			var invalidRect = this._invalidRect;
			invalidRect[0] = left, invalidRect[1] = top, invalidRect[2] = _left, invalidRect[3] = _top;
			invalidRect[4] = left, invalidRect[5] = bottom, invalidRect[6] = _left, invalidRect[7] = _bottom;
			invalidRect[8] = right, invalidRect[9] = top, invalidRect[10] = _right, invalidRect[11] = _top;
			invalidRect[12] = right, invalidRect[13] = bottom, invalidRect[14] = _right, invalidRect[15] = _bottom;
			this.drawStroke(this.interpolateCache, invalidRect);

			for (var i = 0; i < 11; ++i)
				this.strokeCache[i].shift();
			--this.strokeCacheIndex;

			if (this.undoManager)
				this.strokeRect = mergeRect(this.strokeRect, invalidRect);
			return invalidRect;
		}
	}
};