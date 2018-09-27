var Ajax = {
    get: function(url, fn) {
        var obj = new XMLHttpRequest(); 
        obj.open('GET', url, true);
        obj.setRequestHeader("Access-Control-Allow-Origin", "*");
        obj.onreadystatechange = function() {
            if (obj.readyState == 4 && obj.status == 200 || obj.status == 304) {
                fn(obj.responseText);
            }
        };
        obj.send();
    },

    post: function (url, data, fn) {
        var obj = new XMLHttpRequest();
        obj.open("POST", url, true);
        obj.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        obj.setRequestHeader("Access-Control-Allow-Origin", "*");
        obj.onreadystatechange = function() {
            if (obj.readyState == 4 && (obj.status == 200 || obj.status == 304)) {
                fn.call(this, obj.responseText);
            }
        };
        obj.send(data);
    }
};

var Config = function(configUrl = '') {
    this.configUrl = configUrl;
    this.brushSet = [];
    this.keyMap = {};
    this.gesture = {};
    this.configObj = {};
    this.responseText = '';
}

Config.prototype = {
    setConfigUrl: function(url) {
        this.configUrl = url;
    },

    loadFromObject: function(obj) {
        this.brushSet = obj.brushSet;
        this.keyMap = obj.keyMap;
        this.gesture = obj.gesture;
        this.configObj = obj;
    },

    editHotkey: function(key, hotkey) {
        this.keyMap[key] = hotkey;
        console.log(this.keyMap);
        return this;
    },

    switchGestureState: function(key, value) {
        this.gesture[key] = value;
        return this;
    },

    genConfigUrl: function(uid, sessid) {
        // TODO: generate config url
        return this.configUrl;
    },

    loadSettings: function(onSettingsLoaded) {
        var req = new XMLHttpRequest(); 
        req.open('GET', this.configUrl, true);
        var self = this;
        this.done = false;
        req.onreadystatechange = function() {
            if (req.readyState == 4 && req.status == 200 || req.status == 304) {
                if (req.responseText != '') {
                    self.parseConfig(req.responseText);
                    onSettingsLoaded(req.responseText);
                }
            }
            else if (req.status == 404) {
                if (!this.done) {
                    onSettingsLoaded(null);
                    this.done = true;
                }
            }
        };
        req.send();
    },

    parseConfig: function(str) {
        this.configObj = JSON.parse(str);
        this.brushSet = this.configObj['brushSet'];
        this.keyMap = this.configObj['keyMap'];
        this.gesture = this.configObj['gesture'];
        for (var i = 0; i < this.brushSet.length; ++i) {
            if (this.brushSet[i]['hotkey'] != '')
                this.keyMap[this.brushSet[i]['name']] = this.brushSet[i]['hotkey'];
        }
    },

    setId: function(id = '5a8c2a0ce8cb1e0001ef8bee') {
        this.userId = id;
        // mock
        this.saveUrl = 'http://localhost:3000/config/save/' + id;
    },

    saveSettings: function(url = this.saveUrl) {
        var obj = {brushSet: this.brushSet, 'keyMap': this.keyMap, 'gesture': this.gesture};
        Ajax.post(this.saveUrl, 'config=' + JSON.stringify(obj), function(s){});
    }
}