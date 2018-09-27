var CanvasManager = function(canvasName, hotkeyMap, hotkeyCallback, width = -1, height = -1) {
	this.canvasName = canvasName;
	this.canvas = document.getElementById(canvasName);
	this.backgroundColor = [0.157, 0.157, 0.157, 1.0];
	// webgl 2.0 context, supports chrome and macOS Safari
	this.gl = this.canvas.getContext('webgl2', {antialias: false, stencil: false, depth: false, premultipliedAlpha: false});
	this.layerTree = null;
	this.undoManager = null;

	this.scale = 1.0;
	this.rotation = 0.0;
	this.dx = 0.0;
	this.dy = 0.0;
	this.offsetX = 0;
	this.offsetY = 0;
	this.verticesTexCoord = new Float32Array([-1.0, 1.0, 0.0, 1.0,
											-1.0, -1.0, 0.0, 0.0,
											1.0, 1.0, 1.0, 1.0,
											1.0, -1.0, 1.0, 0.0]);

	this.pointerEventListener = new PointerEventListener();
	this.pointerDynamics = new PointerInputDynamicsModel();
	this.keyboardInput = new KeyboardInput();

	this.pointerEventListener.initEventListener(canvasName);
	this.pointerDynamics.init();
	this.keyboardInput.initEventListener(canvasName);
	this.keyboardInput.initKeyMapping(hotkeyMap, hotkeyCallback);

	this.pixelRatio = window.devicePixelRatio || 1;
	if (width < 0) {
		width = window.screen.width * this.pixelRatio;
		height = window.screen.height * this.pixelRatio;
	}
	this.canvas.width = width;
	this.canvas.height = height;
	this.canvas.style.width = (width / this.pixelRatio).toString() + 'px';
	this.canvas.style.height = (height / this.pixelRatio).toString() + 'px';
	this.width = width;
	this.height = height;
	this.radius = -1;
	this.contourPolygon = 18;
	this._lineBuffer = new Float32Array(512);
	this._arrayBuffer = new Float32Array(512);

	this.controlZoom = true;
	this.controlRotation = true;
}

CanvasManager.prototype = {
	drawLine: function(pt, len, color) {
		this.updateDisplay();
		var gl = this.gl;
		this.layerTree.shaderSet.selectShader('util', 0);
		var vertexBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, pt, gl.STATIC_DRAW, 0, len * 2);
		var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
		gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(a_Position);
		var u_Color = gl.getUniformLocation(gl.program, 'u_Color');
		gl.uniform4fv(u_Color, color);
		gl.drawArrays(gl.LINE_STRIP, 0, len);
	},

	updateDisplay: function(curX = null, curY = null, update = true) {
		var layerTree = this.layerTree;
		var gl = this.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, this.width, this.height);
		gl.clearColor(this.backgroundColor[0], this.backgroundColor[1], this.backgroundColor[2], this.backgroundColor[3]);
		gl.clear(gl.COLOR_BUFFER_BIT);
		if (update)
			layerTree.copyFromTexture(layerTree.root.texture[layerTree.root.primaryTexture], this.verticesTexCoord);
		if (curX) {
			var x = curX * this.pixelRatio / this.width * 2 - 1.0;
			var y = 1.0 - curY * this.pixelRatio / this.height * 2;
			this.brushContour(x, y);
		}
	},

	brushContour: function(x, y) {
		var s = this.contourPolygon * 2;
		for (var i = 0; i < s; ++i) {
			this._arrayBuffer[i] = this._lineBuffer[i] * this.scale + x;
			++i;
			this._arrayBuffer[i] = this._lineBuffer[i] * this.scale + y;
		}
		this._arrayBuffer[s++] = this._arrayBuffer[0];
		this._arrayBuffer[s] = this._arrayBuffer[1];
		this.drawLine(this._arrayBuffer, this.contourPolygon + 1, [0.0, 0.0, 0.0, 1.0]);
	},

	setRadius: function(radius, polygon = 24) {
		this.radius = radius;
		this.contourPolygon = polygon;
		var step = 2 * Math.PI / polygon;
		var xFactor = this.height / this.width;// * (this.layerTree.canvasWidth / this.layerTree.canvasHeight);
		var yFactor = 1;//(this.layerTree.canvasWidth / this.layerTree.canvasHeight);
		var theta = 0.0;
		var s = polygon * 2;
		for (var i = 0; i < s; ++i) {
			this._lineBuffer[i++] = Math.cos(theta) * xFactor * radius;
			this._lineBuffer[i] = Math.sin(theta) * yFactor * radius;
			theta += step;
		}
	},

	getPixelRadius: function(r = this.radius) {
		return r / 2 / this.layerTree.canvasHeight * this.height * this.layerTree.canvasWidth * this.scale;
	},

	getNormalizedRadius: function(r) {
		return r * 2 * this.layerTree.canvasHeight / this.height / this.layerTree.canvasWidth / this.scale;
	},

	adjustPerspective: function(dx = null, dy = null, ds = null, dtheta = null) {
		if (dx == null) dx = this.pointerDynamics.deltaTouchCenterX * this.pixelRatio;
		if (dy == null) dy = this.pointerDynamics.deltaTouchCenterY * this.pixelRatio;
		if (ds == null) ds = 4 * this.pointerDynamics.centralStretch /
							(this.height < this.width ? this.width : this.height);
		if (dtheta == null) dtheta = this.pointerDynamics.rotationAngle;
		this.offsetX += dx;
		this.offsetY -= dy;
		this.dx = this.offsetX / this.width * 2;
		this.dy = this.offsetY / this.height * 2;
		if (this.controlRotation) {
			this.rotation -= dtheta;
			if (isNaN(this.rotation))
				this.rotation = 0;
		}
		if (this.controlZoom) {
			this.scale += ds;
			if (this.scale < 0.25) this.scale = 0.25;
			else if (this.scale > 8.0) this.scale = 8.0;
		}
		var x = (this.layerTree.canvasWidth / this.width) * this.scale;
		var y = (this.layerTree.canvasHeight / this.height) * this.scale;
		var w0 = this.layerTree.canvasWidth, h0 = this.layerTree.canvasHeight;
		var W = this.width, H = this.height;
		var transform = function (x, y, s, theta) {
			x *= s, y *= s;
			var x1 = x * Math.cos(theta) - y * Math.sin(theta);
			var y1 = x * Math.sin(theta) + y * Math.cos(theta);
			x1 /= W / 2, y1 /= H / 2;
			return {x: x1, y: y1};
		}
		var c1 = transform(w0 / 2, h0 / 2, this.scale, this.rotation);
		var c2 = transform(-w0 / 2, h0 / 2, this.scale, this.rotation);
		this.verticesTexCoord[0] = c2.x + this.dx, this.verticesTexCoord[1] = c2.y + this.dy,
		this.verticesTexCoord[2] = 0.0, this.verticesTexCoord[3] = 1.0,
		this.verticesTexCoord[4] = -c1.x + this.dx, this.verticesTexCoord[5] = -c1.y + this.dy,
		this.verticesTexCoord[6] = 0.0, this.verticesTexCoord[7] = 0.0,
		this.verticesTexCoord[8] = c1.x + this.dx, this.verticesTexCoord[9] = c1.y + this.dy,
		this.verticesTexCoord[10] = 1.0, this.verticesTexCoord[11] = 1.0,
		this.verticesTexCoord[12] = -c2.x + this.dx, this.verticesTexCoord[13] = -c2.y + this.dy,
		this.verticesTexCoord[14] = 1.0, this.verticesTexCoord[15] = 0.0;
	},

	originalView: function(scale = 1.0) {
		this.scale = scale, this.dx = 0, this.dy = 0, this.rotation = 0, this.offsetX = 0, this.offsetY = 0;
		var x = this.layerTree.canvasWidth / this.width * this.scale;
		var y = this.layerTree.canvasHeight / this.height * this.scale;
		this.verticesTexCoord[0] = -x, this.verticesTexCoord[1] = y, this.verticesTexCoord[2] = 0.0, this.verticesTexCoord[3] = 1.0,
		this.verticesTexCoord[4] = -x, this.verticesTexCoord[5] = -y, this.verticesTexCoord[6] = 0.0, this.verticesTexCoord[7] = 0.0,
		this.verticesTexCoord[8] = x, this.verticesTexCoord[9] = y, this.verticesTexCoord[10] = 1.0, this.verticesTexCoord[11] = 1.0,
		this.verticesTexCoord[12] = x, this.verticesTexCoord[13] = -y, this.verticesTexCoord[14] = 1.0, this.verticesTexCoord[15] = 0.0;
	},

	setPointerStateUpdateCallback: function(onUserInput) {
		this.pointerEventListener.setPointerStateUpdateCallback(onUserInput);
	},

	normalizeCoord: function(x, y, pixelCoord = false) {
		var Ox = this.offsetX;
		var Oy = this.offsetY;
		var x_ = (x * this.pixelRatio - this.width / 2) - Ox;
		var y_ = (this.height / 2 - y * this.pixelRatio) - Oy;
		var _x = Math.cos(-this.rotation) * x_ - Math.sin(-this.rotation) * y_;
		var _y = Math.sin(-this.rotation) * x_ + Math.cos(-this.rotation) * y_;
		x_ = _x / (this.scale * this.layerTree.canvasWidth / 2);
		y_ = _y / (this.scale * this.layerTree.canvasHeight / 2);
		if (pixelCoord) {
			x_ = (1.0 + x_) * this.layerTree.canvasWidth / 2;
			y_ = (1.0 - y_) * this.layerTree.canvasHeight / 2;
		}
		return {'x': x_, 'y': y_};
	},

	pixelCoord: function(x, y) {
		x *= (this.scale * this.layerTree.canvasWidth / 2);
		y *= (this.scale * this.layerTree.canvasHeight / 2);
		var _x = Math.cos(this.rotation) * x - Math.sin(this.rotation) * y;
		var _y = Math.sin(this.rotation) * x + Math.cos(this.rotation) * y;
		var x_ = _x + this.offsetX + this.width / 2;
		var y_ = this.height / 2 - _y - this.offsetY;
		return {x: x_, y: y_};
	},

	setOnKeyUp: function(cb) {
		this.keyboardInput.setOnKeyUp(cb);
	},

	mapPointerOffsetToCoord: function(pointerInputModel) {

	},

	updatePointerInput(input) {
		this.pointerDynamics.updatePointerInput(input);
		this.canvas.focus();
	},
}