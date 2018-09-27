function symmetric(x0, x1) {
	return 2 * x0 - x1;
}

function linearInterpolate(start, end, n, dest) {
	var step = (end - start) / n;
	var t = start;
	for (var i = 0; i < n; ++i) {
		dest[i] = t;
		t += step;
	}
}

function mergeRect(a, b) {
	if (!a)
		return b;
	if (!b)
		return a;
	var left = a[0] < b[0] ? a[0] : b[0];
	var _left = a[2] < b[2] ? a[2] : b[2];
	var right = a[8] > b[8] ? a[8] : b[8];
	var _right = a[10] > b[10] ? a[10] : b[10];
	var bottom = a[5] < b[5] ? a[5] : b[5];
	var _bottom = a[7] < b[7] ? a[7] : b[7];
	var top = a[1] > b[1] ? a[1] : b[1];
	var _top = a[3] > b[3] ? a[3] : b[3];
	return new Float32Array([
		left, top, _left, _top,
		left, bottom, _left, _bottom,
		right, top, _right, _top,
		right, bottom, _right, _bottom]);
}

function genGaussianKernel(size, sigma) {
	var center = Math.floor(size / 2);
	var sum = 0;
	var kernel = new Array(size * size);
	for (var i = 0; i < size; ++i) {
		for (var j = 0; j < size; ++j) {
			var t = (1 / (2 * Math.PI * sigma * sigma))
									* Math.exp(-((i - center) * (i - center)
												+ (j - center) * (j - center)) / (2 * sigma * sigma));
			kernel[i * size + j] = t;
			sum += t;
		}
	}
	var tot = size * size;
	for (var i = 0; i < tot; ++i)
		kernel[i] /= sum;
	return kernel;
}

var CoordTransform = function(x = 0, y = 0, rotation = 0, scale = 1.0, dx = 0, dy = 0) {
	this.setParameter(x, y, rotation, scale, dx, dy);
}

CoordTransform.prototype = {
	setParameter: function(x, y, rotation, scale, dx, dy) {
		this.x = x;
		this.y = y;
		this.rotation = rotation;
		this.scale = scale;
		this.dx = dx;
		this.dy = dy;
	},

	setCanvasMetrics: function(width, height, devicePixelRatio = 1) {
		this.width = width / devicePixelRatio;
		this.height = height / devicePixelRatio;
		this.devicePixelRatio = devicePixelRatio;
	},

	canvasCoord2GL: function(x, y) {
		this.x = x / this.width - 1.0;
		this.y = 1.0 - y / this.height;
	},

	glCoord2Canvas: function(x, y) {
		this.x = (x + 1.0) * this.width;
		this.y = (1.0 - y) * this.height;
	},

	yieldX: function() {
		this.x_prime = this.scale * (this.x * Math.cos(this.rotation) - this.y * Math.sin(this.rotation)) + this.dx;
		return this.x_prime;
	},

	yieldY: function() {
		this.y_prime = this.scale * (this.x * Math.sin(this.rotation) + this.y * Math.cos(this.rotation)) + this.dy;
		return this.y_prime;
	}
};

var CatmullRomSpline = function(max) {
	this.x = new Float32Array(max);
	this.y = new Float32Array(max);
	this.length = 0;
}

CatmullRomSpline.prototype = {
	interpolate: function(x, y = null, n, destX = this.x, destY = this.y, start = 1, end = 2, alpha = 0.5) {
		// interpolate between p1 and p2
		var t = new Array(4);
		t[0] = 0;
		for (var i = 0; i < 4; ++i) {
			if (y != null)
				t[i + 1] = t[i] + Math.pow((x[i + 1] - x[i]) * (x[i + 1] - x[i]) + (y[i + 1] - y[i]) * (y[i + 1] - y[i]), alpha / 2.0);
			else t[i + 1] = t[i] + Math.pow((x[i + 1] - x[i]) * (x[i + 1] - x[i]) * 2, alpha / 2.0);
		}
		var step = (t[end] - t[start]) / n;
		var v = t[start];
		var Ax = new Float32Array(3);
		var Bx = new Float32Array(2);
		var Ay = new Float32Array(3);
		var By = new Float32Array(2);
		var c1, c2;
		for (var i = 0; i < n; ++i) {
			for (var j = 0; j < 3; ++j) {
				c1 = (t[j + 1] - v) / (t[j + 1] - t[j]), c2 = (v - t[j]) / (t[j + 1] - t[j]);
				Ax[j] = c1 * x[j] + c2 * x[j + 1];
				if (y != null)
					Ay[j] = c1 * y[j] + c2 * y[j + 1];
			}
			for (var j = 0; j < 2; ++j) {
				c1 = (t[j + 2] - v) / (t[j + 2] - t[j]), c2 = (v - t[j]) / (t[j + 2] - t[j]);
				Bx[j] = c1 * Ax[j] + c2 * Ax[j + 1];
				if (y != null)
					By[j] = c1 * Ay[j] + c2 * Ay[j + 1];
			}
			destX[i] = ((t[2] - v) * Bx[0] + (v - t[1]) * Bx[1]) / (t[2] - t[1]);
			if (y != null)
				destY[i] = ((t[2] - v) * By[0] + (v - t[1]) * By[1]) / (t[2] - t[1]);
			v += step;
		}
		this.length = n;
	}
};