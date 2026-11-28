/* ============================================================
   植物智能识别系统 · 统一前端交互
   - 真实模型信息徽章
   - 上传（点击 / 拖拽）与加载态
   - 概率柱状图（ECharts，带动画）
   - Toast、图片淡入、数字滚动、粒子背景
   ============================================================ */
(function (global) {
    'use strict';

    var reduceMotion = false;
    try {
        reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) { /* ignore */ }

    function $(selector, root) {
        return (root || document).querySelector(selector);
    }

    function $all(selector, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(selector));
    }

    function escapeHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    /* -------------------- Toast -------------------- */
    function toastHost() {
        var host = $('.plant-toast-host');
        if (!host) {
            host = document.createElement('div');
            host.className = 'plant-toast-host';
            document.body.appendChild(host);
        }
        return host;
    }

    function toast(message, type, timeout) {
        if (!message) return;
        type = type || 'info';
        var el = document.createElement('div');
        el.className = 'plant-toast is-' + type;
        var icon = type === 'success' ? 'bi-check-circle-fill'
            : (type === 'error' ? 'bi-exclamation-triangle-fill' : 'bi-info-circle-fill');
        el.innerHTML = '<i class="bi ' + icon + '"></i><span>' + escapeHtml(message) + '</span>';
        toastHost().appendChild(el);

        var life = timeout || (type === 'error' ? 6000 : 3600);
        setTimeout(function () {
            el.classList.add('is-leaving');
            setTimeout(function () {
                if (el.parentNode) el.parentNode.removeChild(el);
            }, 320);
        }, life);
    }

    /* -------------------- 登录检查 -------------------- */
    function checkLogin() {
        if (!global.fetch) return;
        fetch('/check_login', { credentials: 'same-origin', cache: 'no-store' })
            .then(function (res) { return res.json(); })
            .then(function (data) {
                if (data && data.login === false) {
                    global.location.replace('/');
                }
            })
            .catch(function () { /* 网络异常时交给后端 401 兜底 */ });
    }

    /* -------------------- 按钮加载态 -------------------- */
    function setLoading(button, loading, label) {
        if (!button) return;
        if (loading) {
            if (!button.getAttribute('data-plant-original')) {
                button.setAttribute('data-plant-original', button.innerHTML);
            }
            button.disabled = true;
            button.classList.add('is-loading');
            button.innerHTML = '<span class="plant-spinner"></span>' + escapeHtml(label || '处理中…');
        } else {
            button.disabled = false;
            button.classList.remove('is-loading');
            var original = button.getAttribute('data-plant-original');
            if (original) {
                button.innerHTML = original;
                button.removeAttribute('data-plant-original');
            }
        }
    }

    /* -------------------- 非本模型样本提示 -------------------- */
    function bannerHost() {
        var host = $('.ood-banner-host');
        if (host) return host;
        host = document.createElement('div');
        host.className = 'ood-banner-host';
        var card = $('.upload-card');
        if (card && card.parentNode) {
            card.parentNode.insertBefore(host, card.nextSibling);
        } else {
            document.body.insertBefore(host, document.body.firstChild);
        }
        return host;
    }

    function clearBanner() {
        var host = $('.ood-banner-host');
        if (host) host.innerHTML = '';
    }

    function listHtml(items) {
        return '<ul>' + (items || []).map(function (item) {
            return '<li>' + escapeHtml(item) + '</li>';
        }).join('') + '</ul>';
    }

    /* 后端判定「不是本模型样本」时调用：清掉上一次的结果，明确告诉用户为什么 */
    function showRejection(data) {
        data = data || {};
        var host = bannerHost();
        var reasons = data.reasons || [];
        var hints = data.hints || [];

        var html = '<div class="ood-banner is-reject plant-enter">';
        html += '<div class="ood-banner-head"><i class="bi bi-shield-exclamation"></i><span>' +
            escapeHtml(data.message || '这张图片无法被本模型识别') + '</span></div>';
        if (reasons.length) {
            html += '<div class="ood-banner-body"><strong>原因</strong>' + listHtml(reasons) + '</div>';
        }
        if (hints.length) {
            html += '<div class="ood-banner-body"><strong>建议上传</strong>' + listHtml(hints) + '</div>';
        }
        html += '</div>';
        host.innerHTML = html;

        // 清掉上一次的预测结果，避免用户把旧结果当成这张图的结论
        var title = $('#predict_class');
        if (title) title.innerHTML = '<span style="color:#c0392b;">无法识别该图片</span>';

        var chart = $('#main1');
        if (chart) {
            if (global.echarts) {
                var instance = global.echarts.getInstanceByDom(chart);
                if (instance) instance.dispose();
            }
            chart.innerHTML = '<div class="chart-loading"><i class="bi bi-shield-exclamation"></i>' +
                ' 未输出概率分布</div>';
        }

        var resultCard = $('#result-card');
        if (resultCard) resultCard.style.display = 'none';

        var count = $('#grape-count');
        if (count) count.textContent = '0';

        var report = $('#report');
        if (report) report.innerHTML = '';
        var treatment = $('#treatment');
        if (treatment) treatment.innerHTML = '';
        var alertSlot = $('#alert-slot');
        if (alertSlot) alertSlot.innerHTML = '';

        if (host.scrollIntoView) {
            host.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
        }
        return true;
    }

    /* 后端判定「疑似」：照常给结果，但顶部挂一条可关闭的提醒 */
    function showWarning(warning) {
        warning = warning || {};
        var host = bannerHost();
        var html = '<div class="ood-banner is-warn plant-enter">';
        html += '<button type="button" class="ood-banner-close" aria-label="关闭">&times;</button>';
        html += '<div class="ood-banner-head"><i class="bi bi-exclamation-triangle"></i><span>' +
            escapeHtml(warning.message || '这张图片可能不是本模型的典型样本，结果仅供参考。') + '</span></div>';
        if ((warning.reasons || []).length) {
            html += '<div class="ood-banner-body"><strong>参考信息</strong>' + listHtml(warning.reasons) + '</div>';
        }
        html += '</div>';
        host.innerHTML = html;

        var close = $('.ood-banner-close', host);
        if (close) {
            close.addEventListener('click', function () { clearBanner(); });
        }
        return true;
    }

    /* -------------------- 上传 + 拖拽 + 请求 -------------------- */
    function bindUpload(options) {
        options = options || {};
        var endpoint = options.endpoint;
        var input = $(options.input || '#i-file');
        var button = $(options.submit || '#submit');
        var card = $(options.drop || '.upload-card');
        var label = $(options.label || '#location');
        var preview = options.preview ? $(options.preview) : null;

        if (!endpoint || !input || !button) {
            console.warn('[PlantUI] bindUpload: 缺少 endpoint / input / submit');
            return;
        }

        var selected = null;

        function selectFile(file) {
            if (!file) return;
            if (!/^image\//.test(file.type) && !/\.(png|jpe?g)$/i.test(file.name)) {
                toast('仅支持 PNG、JPG、JPEG 图片', 'error');
                return;
            }
            if (file.size > 20 * 1024 * 1024) {
                toast('图片超过 20MB，请压缩后再上传', 'error');
                return;
            }
            selected = file;
            if (label) {
                label.value = file.name;
                label.classList.add('file-selected');
            }
            if (preview) {
                preview.src = URL.createObjectURL(file);
                showImage(preview);
            }
        }

        input.addEventListener('change', function () {
            selectFile(input.files && input.files[0]);
        });

        if (card) {
            ['dragenter', 'dragover'].forEach(function (name) {
                card.addEventListener(name, function (event) {
                    event.preventDefault();
                    event.stopPropagation();
                    card.classList.add('dragover');
                });
            });
            ['dragleave', 'drop'].forEach(function (name) {
                card.addEventListener(name, function (event) {
                    event.preventDefault();
                    event.stopPropagation();
                    if (name === 'dragleave' && card.contains(event.relatedTarget)) return;
                    card.classList.remove('dragover');
                });
            });
            card.addEventListener('drop', function (event) {
                var files = event.dataTransfer && event.dataTransfer.files;
                if (files && files.length) {
                    try {
                        input.files = files;
                    } catch (e) { /* 某些浏览器只读，下面直接使用 selected */ }
                    selectFile(files[0]);
                }
            });
        }

        async function submit() {
            var file = selected || (input.files && input.files[0]);
            if (!file) {
                toast('请先选择要识别的图片', 'error');
                if (options.onMissingFile) options.onMissingFile();
                return;
            }
            if (button.disabled) return;

            setLoading(button, true, options.loadingText || '识别中…');
            var controller = new AbortController();
            var timer = setTimeout(function () { controller.abort(); }, options.timeout || 90000);
            var started = performance.now();

            try {
                var formData = new FormData();
                formData.append('file', file, file.name);
                var response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData,
                    credentials: 'same-origin',
                    signal: controller.signal
                });

                var text = await response.text();
                var data;
                try {
                    data = JSON.parse(text);
                } catch (parseError) {
                    throw new Error(response.ok ? '服务器返回数据异常' : ('请求失败（HTTP ' + response.status + '）'));
                }

                if (!response.ok || data.error) {
                    throw new Error(data.error || ('请求失败（HTTP ' + response.status + '）'));
                }

                var elapsed = Math.round(performance.now() - started);

                // 后端判定这张图不是本模型能识别的样本：不渲染任何预测结果
                if (data.valid === false) {
                    showRejection(data);
                    if (options.onInvalid) options.onInvalid(data);
                    return;
                }

                if (data.warning) {
                    showWarning(data.warning);
                } else {
                    clearBanner();
                }

                options.onSuccess(data, elapsed);
                if (options.successToast !== false) {
                    toast((options.successText || '识别完成') + ' · ' + elapsed + ' ms', 'success');
                }
            } catch (error) {
                var message = error && error.name === 'AbortError'
                    ? '请求超时，请稍后重试'
                    : (error && error.message ? error.message : '请求失败，请稍后重试');
                toast(message, 'error', 6500);
                if (options.onError) options.onError(error);
            } finally {
                clearTimeout(timer);
                setLoading(button, false);
            }
        }

        button.addEventListener('click', function (event) {
            event.preventDefault();
            submit();
        });
    }

    /* -------------------- 概率柱状图 -------------------- */
    function wrapLabel(value) {
        var text = String(value == null ? '' : value);
        if (text.length <= 8) return text;
        if (/[A-Za-z]/.test(text)) {
            var words = text.split(/\s+/);
            var lines = [];
            var line = '';
            words.forEach(function (word) {
                if ((line + ' ' + word).trim().length > 11) {
                    if (line) lines.push(line.trim());
                    line = word;
                } else {
                    line += ' ' + word;
                }
            });
            if (line.trim()) lines.push(line.trim());
            return lines.join('\n');
        }
        return text.replace(/(.{6})/g, '$1\n').replace(/\n$/, '');
    }

    function probabilityChart(dom, config) {
        if (!dom) return null;
        config = config || {};

        if (!global.echarts) {
            dom.innerHTML = '<div class="chart-loading"><i class="bi bi-exclamation-circle"></i> 图表组件加载失败</div>';
            return null;
        }

        var labels = (config.labels || []).slice();
        var labelsEn = (config.labelsEn || []).slice();
        var values = (config.values || config.probs || []).map(function (value) {
            return Number(value) || 0;
        });
        if (!labels.length || !values.length) {
            dom.innerHTML = '<div class="chart-loading"><i class="bi bi-inbox"></i> 暂无概率数据</div>';
            return null;
        }

        var max = Math.max.apply(null, values);
        var topIndex = typeof config.topIndex === 'number' ? config.topIndex : values.indexOf(max);
        var hasSubtitle = !!config.subtitle;
        var palette = ['#667eea', '#4facfe', '#11998e', '#f5576c', '#f093fb', '#764ba2', '#00c2ff', '#38ef7d', '#fa709a', '#f6d365'];

        var seriesData = values.map(function (value, index) {
            var item = {
                value: Number(value.toFixed(2)),
                itemStyle: {
                    borderRadius: [10, 10, 4, 4],
                    color: index === topIndex
                        ? new global.echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#5b6ee1' },
                            { offset: 1, color: '#8b5fbf' }
                        ])
                        : palette[index % palette.length],
                    shadowBlur: index === topIndex ? 14 : 0,
                    shadowColor: index === topIndex ? 'rgba(102, 126, 234, 0.45)' : 'transparent'
                }
            };
            return item;
        });

        var chart = global.echarts.getInstanceByDom(dom);
        if (!chart) {
            dom.innerHTML = '';
            chart = global.echarts.init(dom, null, { renderer: 'canvas' });
            dom.__plantChart = chart;
        }

        // 卡片高度随图片/窗口变化时，图表自动铺满剩余空间
        if (!dom.__plantResizeObserver && global.ResizeObserver) {
            dom.__plantResizeObserver = new global.ResizeObserver(function () {
                chart.resize();
            });
            dom.__plantResizeObserver.observe(dom);
        }

        var option = {
            backgroundColor: 'transparent',
            animation: !reduceMotion,
            animationDuration: 780,
            animationEasing: 'cubicOut',
            animationDurationUpdate: 520,
            title: {
                top: 4,
                left: 'center',
                text: config.title || '模型预测概率分布',
                subtext: config.subtitle || '',
                itemGap: 6,
                textStyle: { fontSize: 17, fontWeight: 700, color: '#2c3e50' },
                subtextStyle: { fontSize: 12, color: '#8a94a6' }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(102,126,234,0.06)' } },
                backgroundColor: 'rgba(255,255,255,0.98)',
                borderColor: 'rgba(102,126,234,0.25)',
                borderWidth: 1,
                textStyle: { color: '#2c3e50', fontSize: 12 },
                formatter: function (params) {
                    var item = params[0];
                    var index = item.dataIndex;
                    var en = labelsEn[index] ? ' <span style="color:#8a94a6">' + escapeHtml(labelsEn[index]) + '</span>' : '';
                    return '<b>' + escapeHtml(labels[index]) + '</b>' + en +
                        '<br/>预测概率：<b style="color:#667eea">' + item.value + '%</b>';
                }
            },
            grid: {
                left: '3%',
                right: '3%',
                bottom: '4%',
                top: (labels.length > 6 ? 74 : 66) + (hasSubtitle ? 18 : 0),
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: labels,
                axisTick: { show: false },
                axisLine: { lineStyle: { color: 'rgba(44,62,80,0.15)' } },
                axisLabel: {
                    interval: 0,
                    fontSize: 11,
                    color: '#5b6472',
                    lineHeight: 15,
                    rotate: typeof config.rotate === 'number' ? config.rotate : (labels.length > 6 ? 32 : 0),
                    formatter: wrapLabel
                }
            },
            yAxis: {
                type: 'value',
                name: '概率 (%)',
                nameTextStyle: { color: '#8a94a6', fontSize: 12, padding: [0, 0, 6, 0] },
                max: 100,
                axisLabel: { color: '#8a94a6', fontSize: 11, formatter: '{value}%' },
                splitLine: { lineStyle: { color: 'rgba(102,126,234,0.12)', type: 'dashed' } }
            },
            series: [{
                type: 'bar',
                name: '预测概率',
                data: seriesData,
                barMaxWidth: 46,
                emphasis: {
                    focus: 'series',
                    itemStyle: { shadowBlur: 18, shadowColor: 'rgba(102,126,234,0.5)' }
                },
                label: {
                    show: true,
                    position: 'top',
                    distance: 6,
                    formatter: '{c}%',
                    fontSize: 11,
                    fontWeight: 700,
                    color: '#2c3e50'
                }
            }]
        };

        chart.setOption(option, true);
        chart.resize();

        if (!dom.__plantResizeBound) {
            dom.__plantResizeBound = true;
            global.addEventListener('resize', function () { chart.resize(); }, { passive: true });
        }
        return chart;
    }

    function resizeChart(target) {
        if (!global.echarts) return;
        var dom = typeof target === 'string' ? $(target) : target;
        if (!dom) return;
        var chart = global.echarts.getInstanceByDom(dom);
        if (chart) chart.resize();
    }

    /* -------------------- 模型徽章 -------------------- */
    function modelBadge(model) {
        if (!model) return '';
        var architecture = escapeHtml(model.architecture || model.name || '深度学习模型');
        var device = String(model.device || '').split('·')[0].trim() || 'CPU';
        var html = '<span class="model-chip"><i class="bi bi-diagram-3"></i>' + architecture + '</span>';
        html += '<span class="model-chip"><i class="bi bi-cpu"></i>' + escapeHtml(device) + '</span>';
        if (model.accuracy_text && model.accuracy_text !== '—') {
            html += '<span class="model-chip is-success"><i class="bi bi-bullseye"></i>测试准确率 ' + escapeHtml(model.accuracy_text) + '</span>';
        }
        if (model.elapsed_ms != null) {
            html += '<span class="model-chip is-success"><i class="bi bi-stopwatch"></i>' + Math.round(model.elapsed_ms) + ' ms</span>';
        }
        html += '<span class="model-chip is-success"><i class="bi bi-patch-check"></i>真实模型推理</span>';
        return html;
    }

    function renderModelBadge(target, model) {
        var el = typeof target === 'string' ? $(target) : target;
        if (el) el.innerHTML = modelBadge(model);
    }

    /* -------------------- 视觉辅助 -------------------- */
    function showImage(image) {
        if (!image) return;
        image.style.display = image.tagName === 'IMG' ? 'inline-block' : '';
        image.classList.remove('plant-img-in');
        void image.offsetWidth;
        image.classList.add('plant-img-in');
    }

    function enter(elements, baseDelay) {
        $all(elements && elements.length !== undefined ? elements.join(',') : (elements || '.result-card'))
            .forEach(function (element, index) {
                element.classList.remove('plant-enter');
                void element.offsetWidth;
                element.style.setProperty('--plant-delay', ((baseDelay || 0) + index * 70) + 'ms');
                element.classList.add('plant-enter');
            });
    }

    function countUp(element, target, options) {
        if (!element) return;
        options = options || {};
        var decimals = options.decimals || 0;
        var suffix = options.suffix || '';
        var duration = reduceMotion ? 0 : (options.duration || 900);
        var start = performance.now();
        var from = 0;

        function frame(now) {
            var progress = duration === 0 ? 1 : Math.min(1, (now - start) / duration);
            var eased = 1 - Math.pow(1 - progress, 3);
            var value = from + (target - from) * eased;
            element.textContent = value.toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* -------------------- 粒子背景 -------------------- */
    function initParticles(options) {
        options = options || {};
        if (reduceMotion) return;
        var container = $(options.container || '#particles');
        if (!container || container.getAttribute('data-plant-ready') === '1') return;
        container.setAttribute('data-plant-ready', '1');

        var count = options.count || 18;
        for (var i = 0; i < count; i++) {
            var particle = document.createElement('div');
            particle.className = 'particle';
            var size = (Math.random() * 4 + 2).toFixed(1);
            var duration = (Math.random() * 16 + 18).toFixed(1);
            var delay = (-Math.random() * 20).toFixed(1);
            var drift = (Math.random() * 120 - 60).toFixed(0);
            var opacity = (Math.random() * 0.45 + 0.25).toFixed(2);

            particle.style.cssText =
                'left:' + (Math.random() * 100).toFixed(2) + '%;' +
                'top:' + (85 + Math.random() * 20).toFixed(1) + '%;' +
                'width:' + size + 'px;height:' + size + 'px;' +
                'opacity:' + opacity + ';' +
                '--p-drift:' + drift + 'px;' +
                '--p-opacity:' + opacity + ';' +
                'animation: plant-particle-float ' + duration + 's linear ' + delay + 's infinite;';
            container.appendChild(particle);
        }
    }

    /* -------------------- 初始化 -------------------- */
    document.addEventListener('DOMContentLoaded', function () {
        if ($('#particles')) initParticles();
    });

    global.PlantUI = {
        toast: toast,
        checkLogin: checkLogin,
        setLoading: setLoading,
        bindUpload: bindUpload,
        probabilityChart: probabilityChart,
        resizeChart: resizeChart,
        modelBadge: modelBadge,
        renderModelBadge: renderModelBadge,
        showImage: showImage,
        enter: enter,
        countUp: countUp,
        initParticles: initParticles,
        showRejection: showRejection,
        showWarning: showWarning,
        clearBanner: clearBanner,
        escapeHtml: escapeHtml,
        wrapLabel: wrapLabel
    };
})(window);
