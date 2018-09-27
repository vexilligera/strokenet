// Pointer input model
var PointerInputModel = function() {
	// { identifier, offsetX, offsetY, time, pressure }
	this.touches = [];
	this.touchCenterX = -1;
	this.touchCenterY = -1;
	this.averageCentralDistance = 0;
	this.primaryTouchIndex = -1;
	// pen model
	this.penId = -1;
	this.penOffsetX = -1;
	this.penOffsetY = -1;
	this.penPressure = -1;
	this.penTiltX = 0;
	this.penTltY = 0;
	this.isPenDown = false;
	this.penTime = -1;
	// mouse model
	this.mouseOffsetX = -1;
	this.mouseOffsetY = -1;
	this.isLeftDown = 0;
	this.isRightDown = 0;
	this.isMiddleDown = 0;
	this.dxScroll = 0;
	this.dyScroll = 0;
}

PointerInputModel.prototype = {
	euclideanDistance: function(x1, y1, x2, y2) {
		return Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	},

	updateAverageCentralDistance: function() {
		var sum = 0;
		for (var i = 0; i < this.touches.length; ++i)
			sum += this.euclideanDistance(this.touches[i].offsetX, this.touches[i].offsetY, this.touchCenterX, this.touchCenterY);
		this.averageCentralDistance = sum / this.touches.length;
	},
	
	addTouch: function(touch) {
		var index = this.touchIndexByIdentifier(touch.identifier);
		if (index == -1)
			this.touches.push(this.copyTouch(touch));
		else
			this.touches.splice(index, 1, this.copyTouch(touch));
		var sumX = 0, sumY = 0;
		for (var i = 0; i < this.touches.length; ++i) {
			sumX += this.touches[i].offsetX;
			sumY += this.touches[i].offsetY;
		}
		if (this.touches.length == 1)
			this.primaryTouchIndex = 0;
		this.touchCenterX = sumX / this.touches.length;
		this.touchCenterY = sumY / this.touches.length;
		this.updateAverageCentralDistance();
	},

	removeTouch: function(id) {
		var index = this.touchIndexByIdentifier(id);
		this.touches.splice(index, 1);
		if (this.touches.length == 0) {
			this.primaryTouchIndex = -1;
			this.touchCenterX = -1;
			this.touchCenterY = -1;
		}
		this.updateAverageCentralDistance();
	},

	touchIndexByIdentifier: function(identifier) {
		for (var i = 0; i < this.touches.length; ++i) {
			var id = this.touches[i].identifier;
			if (id == identifier)
				return i;
		}
		return -1;
	},

	copyTouch: function(touch) {
		var curTime = new Date().getTime();
		return { identifier: touch.identifier, offsetX: touch.offsetX, offsetY: touch.offsetY, time: curTime, pressure: touch.pressure };
	},

	deepCopy: function(Obj) {
		var newObj;   
		if (Obj instanceof Array) {
			newObj = [];
			var i = Obj.length;
			while (i--)
				newObj[i] = this.deepCopy(Obj[i]);
			return newObj;
		}
		else if (Obj instanceof Object) {
			newObj = {};
			for (var k in Obj)
				newObj[k] = this.deepCopy(Obj[k]);
			return newObj;
		}
		else
			return Obj;
    },

    Clone: function() {
    	return this.deepCopy(this);
    }

};

// Event listeners, may be browser dependent
// For Safari pressure should be obtained from touch event
var PointerEventListener = function() {
	this.pointerInputModel = {};
	this.pointerStateUpdateCallback = null;
}

PointerEventListener.prototype = {

	setPointerStateUpdateCallback: function(callback) {
		this.pointerStateUpdateCallback = callback;
	},

	getPointerInputModel: function() {
		this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerOverListener: function(event) {
		//event.preventDefault();
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerEnterListener: function(event) {
		//event.preventDefault();
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerDownListener: function(event) {
		event.preventDefault();
		var touch = new Object();
		switch (event.pointerType) {
		case "mouse":
			if (event.button <= 1)
				this.pointerInputModel.isLeftDown = true;
			if (event.button == 2)
				this.pointerInputModel.isRightDown = true;
			break;
		case "pen":
			this.pointerInputModel.isPenDown = true;
			this.pointerInputModel.penId = event.pointerId;
			this.pointerInputModel.penOffsetX = event.offsetX;
			this.pointerInputModel.penOffsetY = event.offsetY;
			this.pointerInputModel.penPressure = event.pressure;
			this.pointerInputModel.penTiltX = event.tiltX;
			this.pointerInputModel.penTiltY = event.tiltY;
			this.pointerInputModel.penTime = new Date().getTime();
			break;
		case "touch":
			if (!this.pointerInputModel.isPenDown) {
				touch.identifier = event.pointerId;
				touch.offsetX = event.offsetX;
				touch.offsetY = event.offsetY;
				// For Safari on iPad Pro pen event is touch with pressure
				touch.pressure = event.pressure;
				this.pointerInputModel.addTouch.call(this.pointerInputModel, touch);
			}
			break;
		default:
		}
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerMoveListener: function(event) {
		event.preventDefault();
		var touch = new Object();
		switch (event.pointerType) {
		case "mouse":
			this.pointerInputModel.mouseOffsetX = event.offsetX;
			this.pointerInputModel.mouseOffsetY = event.offsetY;
			this.pointerInputModel.penOffsetX = -1;
			this.pointerInputModel.penOffsetY = -1;
			break;
		case "pen":
			this.pointerInputModel.penId = event.pointerId;
			this.pointerInputModel.penOffsetX = event.offsetX;
			this.pointerInputModel.penOffsetY = event.offsetY;
			this.pointerInputModel.penPressure = event.pressure;
			this.pointerInputModel.penTiltX = event.tiltX;
			this.pointerInputModel.penTiltY = event.tiltY;
			this.pointerInputModel.penTime = new Date().getTime(); 
			this.pointerInputModel.mouseOffsetX = -1;
			this.pointerInputModel.mouseOffsetY = -1;
			break;
		case "touch":
			touch.identifier = event.pointerId;
			touch.offsetX = event.offsetX;
			touch.offsetY = event.offsetY;
			// For Safari on iPad Pro pen event is touch with pressure
			touch.pressure = event.pressure;
			this.pointerInputModel.addTouch.call(this.pointerInputModel, touch);
			break;
		default:
		}
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerUpListener: function(event) {
		event.preventDefault();
		switch (event.pointerType) {
		case "mouse":
			if (event.button <= 1)
				this.pointerInputModel.isLeftDown = false;
			if (event.button == 2)
				this.pointerInputModel.isRightDown = false;
			break;
		case "pen":
			this.pointerInputModel.isPenDown = false;
			this.pointerInputModel.penId = -1;
			this.pointerInputModel.penOffsetX = -1;
			this.pointerInputModel.penOffsetY = -1;
			this.pointerInputModel.penPressure = -1;
			this.pointerInputModel.penTiltX = 0;
			this.pointerInputModel.penTiltY = 0;
			break;
		case "touch":
			this.pointerInputModel.removeTouch.call(this.pointerInputModel, event.pointerId);
			break;
		default:
		}
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerCancelListener: function(event) {
		event.preventDefault();
		switch (event.pointerType) {
		case "mouse":
			if (event.buttons == 1)
				this.pointerInputModel.isLeftDown = false;
			if (event.buttons == 2)
				this.pointerInputModel.isRightDown = false;
			break;
		case "pen":
			this.pointerInputModel.isPenDown = false;
			break;
		case "touch":
			this.pointerInputModel.removeTouch.call(this.pointerInputModel, event.pointerId);
			break;
		default:
		}
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerOutListener: function(event) {
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerLeaveListener: function(event) {
		switch (event.pointerType) {
		case "mouse":
			break;
		case "pen":
			break;
		case "touch":
			this.pointerInputModel.removeTouch.call(this.pointerInputModel, event.pointerId);
			break;
		default:
		}
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerGotCaptureListener: function(event) {
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	pointerLostCaptureListener: function(event) {
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	mouseWheel: function(event) {
		this.pointerInputModel.dxScroll += event.deltaX;
		this.pointerInputModel.dyScroll += event.deltaY;
		if (this.pointerStateUpdateCallback != null)
			this.pointerStateUpdateCallback(this.pointerInputModel);
	},

	initEventListener: function(target) {
		// works for Chrome, safari needs alternative methods
		var self = this;
		var e = document.getElementById(target);
		e.addEventListener("pointerover", function(event) {
			self.pointerOverListener.call(self, event);
			self.dxScroll = 0, self.dyScroll = 0;
		}, false);
		e.addEventListener("pointerenter", function(event) {
			self.pointerEnterListener.call(self, event);
		}, false);
		e.addEventListener("pointerdown", function(event) {
			self.pointerDownListener.call(self, event);
		}, false);
		e.addEventListener("pointermove", function(event) {
			self.pointerMoveListener.call(self, event);
		}, false);
		e.addEventListener("pointerup", function(event) {
			self.pointerUpListener.call(self, event);
		}, false);
		e.addEventListener("pointercancel", function(event) {
			self.pointerCancelListener.call(self, event);
		}, false);
		e.addEventListener("pointerleave", function(event) {
			self.pointerLeaveListener.call(self, event);
		}, false);
		e.addEventListener("gotpointercapture", function(event) {
			self.pointerGotCaptureListener.call(self, event);
		}, false);
		e.addEventListener("lostpointercapture", function(event) {
			self.pointerLostCaptureListener.call(self, event);
		}, false);
		e.addEventListener("mousewheel", function(event) {
			self.mouseWheel.call(self, event);
		}, false);
		this.pointerInputModel = new PointerInputModel();
	}

};

// Input dynamics and touch gesture model
var PointerInputDynamicsModel = function() {
	this.previousPointerState = {};
	this.currentPointerState = {};
	// touch dynamics
	// { identifier, deltaX, deltaY, time, deltaPressure }
	this.deltaTouches = [];
	this.deltaTouchCenterX = -1;
	this.deltaTouchCenterY = -1;
	this.rotationAngle = 0.0;
	this.centralStretch = 0.0;
	// pen model
	this.penId = -1;
	this.deltaPenX = -1;
	this.deltaPenY = -1;
	this.deltaPenPressure = -1;
	this.deltaPenTiltX = 0;
	this.deltaPenTiltY = 0;
	this.penVelocityX = -1.0;
	this.penVelocityY = -1.0;
	this.deltaPenTime = 0;
	this.penDist = 0.0;
	// mouse model
	this.deltaMouseX = -1;
	this.deltaMouseY = -1;
	this.isLeftDown = 0;
	this.isRightDown = 0;
	this.isMiddleDown = 0;
	this.dxCumulativeScroll = 0;
	this.dyCumulativeScroll = 0;
}

PointerInputDynamicsModel.prototype = {
	init: function() {
		this.previousPointerState = new PointerInputModel();
		this.currentPointerState = new PointerInputModel();
	},

	updatePointerInput: function(newState) {
		this.previousPointerState = this.deepCopy(this.currentPointerState);
		this.currentPointerState = this.deepCopy(newState);
		// touch dynamics
		this.deltaTouches.splice(0, this.deltaTouches.length);
		for (var i = 0; i < this.previousPointerState.touches.length; ++i) {
			for (var j = 0; j < this.currentPointerState.touches.length; ++j) {
				if (this.previousPointerState.touches[i].identifier == this.currentPointerState.touches[j].identifier) {
					var delta = new Object();
					var cur = this.currentPointerState.touches[j];
					var prev = this.previousPointerState.touches[i];
					delta.identifier = this.previousPointerState.touches[i].identifier;
					delta.deltaX = cur.offsetX - prev.offsetX;
					delta.deltaY = cur.offsetY - prev.offsetY;
					delta.time = cur.time - prev.time;
					delta.pressure = cur.pressure - prev.pressure;
					this.deltaTouches.push(delta);
				}
			}
		}
		var sumX = 0, sumY = 0;
		for (var i = 0; i < this.deltaTouches.length; ++i) {
			sumX += this.deltaTouches[i].deltaX;
			sumY += this.deltaTouches[i].deltaY;
		}
		this.deltaTouchCenterX = sumX / this.deltaTouches.length;
		this.deltaTouchCenterY = sumY / this.deltaTouches.length;
		if (this.previousPointerState.touches.length > 1 && this.currentPointerState.touches.length > 1) {
			var id1 = this.previousPointerState.touches[this.previousPointerState.primaryTouchIndex].identifier;
			var id2 = this.currentPointerState.touches[this.currentPointerState.primaryTouchIndex].identifier;
			if (id1 == id2) {
				var curPrimary = this.currentPointerState.touches[this.currentPointerState.primaryTouchIndex];
				var prevPrimary = this.previousPointerState.touches[this.previousPointerState.primaryTouchIndex];
				var k2 = (curPrimary.offsetY - this.currentPointerState.touchCenterY) / (curPrimary.offsetX - this.currentPointerState.touchCenterX);
				var k1 = (prevPrimary.offsetY - this.previousPointerState.touchCenterY) / (prevPrimary.offsetX - this.previousPointerState.touchCenterX);
				var tan = (k2 - k1) / (1 + k1 * k2);
				this.rotationAngle = Math.atan(tan);
			}
			else this.rotationAngle = 0.0;
		}
		this.centralStretch = this.currentPointerState.averageCentralDistance - this.previousPointerState.averageCentralDistance;
		if (isNaN(this.centralStretch) || this.centralStretch > 50) this.centralStretch = 0.0;
		// pen dynamics
		this.deltaPenX = this.currentPointerState.penOffsetX - this.previousPointerState.penOffsetX;
		this.deltaPenY = this.currentPointerState.penOffsetY - this.previousPointerState.penOffsetY;
		this.deltaPenTime = this.currentPointerState.penTime - this.previousPointerState.penTime;
		this.penVelocityX = this.deltaPenX / this.deltaPenTime;
		this.penVelocityY = this.deltaPenY / this.deltaPenTime;
		// mouse dynamics

	},

	deepCopy: function(Obj) {
		var newObj;   
		if (Obj instanceof Array) {
			newObj = [];
			var i = Obj.length;
			while (i--)
				newObj[i] = this.deepCopy(Obj[i]);
			return newObj;
		}
		else if (Obj instanceof Object) {
			newObj = {};
			for (var k in Obj)
				newObj[k] = this.deepCopy(Obj[k]);
			return newObj;
		}
		else
			return Obj;
    }
};

// keyboard shortcut input
var KeyboardInput = function() {
	this.ctrlPressed = false;
	this.altPressed = false;
	this.shiftPressed = false;
	this.onAlt = 0;
	this.currentKey = '';
	this.currentHotkey = '';
	this.hotkeyMap = {};
	this.callbackMap = {};
	this.onKeyUp = null;
}

KeyboardInput.prototype = {
	initEventListener: function(target) {
		var self = this;
		var e = document.getElementById(target);
		e.addEventListener('keydown', function(event) {
			self.keyDownListener.call(self, event);
		}, true);
		e.addEventListener('keyup', function(event) {
			self.keyUpListener.call(self, event);
		}, true);
	},

	initKeyMapping: function(keyMap, callbackMap) {
		this.hotkeyMap = new Object(keyMap);
		for (var i in this.hotkeyMap) {
			hotkey = this.hotkeyMap[i];
			hotkey = hotkey.toLowerCase();
			var arr = hotkey.split(' ');
			var key = '';
			var ctrl = false, shift = false, alt = false;
			for (var j = 0; j < arr.length; ++j) {
				switch (arr[j]) {
				case 'ctrl':
					ctrl = true;
					break;
				case 'shift':
					shift = true;
					break;
				case 'alt':
					alt = true;
					break;
				default:
					key = arr[j];
					break;
				}
			}
			shortcut = '';
			if (ctrl)
				shortcut += 'ctrl ';
			if (alt)
				shortcut += 'alt ';
			if (shift)
				shortcut += 'shift ';
			if (key != '')
				shortcut += key;
			this.hotkeyMap[i] = shortcut;
		}
		this.callbackMap = new Object(callbackMap);
	},

	keyDownListener: function(event) {
		switch (event.key) {
		case 'Control':
			this.ctrlPressed = true;
			break;
		case 'Alt':
			this.altPressed = true;
			++this.onAlt;
			if (this.onAlt > 2) {
				this.dispatchHotkey('alt ');
			}
			break;
		case 'Shift':
			this.shiftPressed = true;
			break;
		default:
			this.currentKey = event.key.toLowerCase();
			if (this.ctrlPressed) {
				this.currentHotkey += 'ctrl ';
			}
			if (this.altPressed) {
				this.currentHotkey += 'alt ';
			}
			if (this.shiftPressed) {
				this.currentHotkey += 'shift ';
			}
			this.currentHotkey += this.currentKey;
			if (this.currentHotkey.indexOf(' ') < 0)
				this.currentHotkey = this.currentHotkey[0];
			this.dispatchHotkey(this.currentHotkey);
			break;
		}
	},

	keyUpListener: function(event) {
		switch (event.key) {
		case 'Control':
			this.ctrlPressed = false;
			break;
		case 'Alt':
			this.altPressed = false;
			this.onAlt = 0;
			break;
		case 'Shift':
			this.shiftPressed = false;
			break;
		default:
			this.currentKey = '';
			this.currentHotkey = '';
			break;
		}
		if (this.onKeyUp)
			this.onKeyUp();
	},

	dispatchHotkey: function(hotkey) {
		for (var i in this.hotkeyMap) {
			if (this.hotkeyMap[i] == hotkey) {
				if (this.callbackMap[i] != null) {
					this.callbackMap[i](i);
				}
			}
		}
	},

	setOnKeyUp: function(cb) {
		this.onKeyUp = cb;
	}
};