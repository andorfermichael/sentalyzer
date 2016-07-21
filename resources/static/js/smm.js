var mapOptions = {
  zoom: 3,
  center: new google.maps.LatLng(8.881928, 76.592758),
  mapTypeId: google.maps.MapTypeId.ROADMAP
}
var map = new google.maps.Map(document.getElementById("map_canvas"), mapOptions);

var positiveHeatMapData = [];
var negativeHeatMapData = [];

var positiveGradient = [
    'rgba(152, 251, 152, 0)',
    'rgba(135, 234, 135, 1)',
    'rgba(118, 217, 118, 1)',
    'rgba(101, 200, 101, 1)',
    'rgba(84, 183, 84, 1)',
    'rgba(67, 167, 67, 1)',
    'rgba(50, 150, 50, 1)',
    'rgba(33, 133, 33, 1)',
    'rgba(16, 116, 16, 1)',
    'rgba(0, 100, 0, 1)'
];

var negativeGradient = [
    'rgba(255, 192, 203, 0)',
    'rgba(240, 171, 180, 1)',
    'rgba(225, 151, 158, 1)',
    'rgba(211, 131, 136, 1)',
    'rgba(196, 111, 113, 1)',
    'rgba(182, 90, 91, 1)',
    'rgba(167, 70, 69, 1)',
    'rgba(153, 50, 46, 1)',
    'rgba(138, 30, 24, 1)',
    'rgba(124, 10, 2, 1)'
];

var SMM = {
    streamChannel: {
        stopTracking: '#stop-tracking',
        restartTracking: '#restart-tracking',
        stream: null,
        listen: function(kw) {

            if (SMM.stream == null) {
                SMM.stream = io.connect('/stream');
            }

            SMM.stream.on('stream_update', function(data) {
                SMM.streamData.push(data);
            });
            SMM.stream.on('error', function(data) {
                $('#error-msg section').html(data);
                $('#error-msg').slideDown();
                $("#keyword").removeClass('loading');
                SMM.stream.disconnect();
                SMM.stream = null;
                
            });

            $(SMM.streamChannel.stopTracking).click(function() {
                SMM.stream.disconnect();
                SMM.stream = null;
                $("#keyword").removeClass('loading');
                $(this).attr('disabled','disabled');
                return false;
            });

            $(SMM.streamChannel.restartTracking).click(function() {
                SMM.stream.disconnect();
                SMM.stream = null;
                $(this).submit();
            });
            
            SMM.stream.emit('track', kw);
            SMM.streamData.clear();
            SMM.charts.redraw();

            setInterval(function() {
                if(SMM.stream){
                    SMM.stream.emit('ping')
                }
            }, 2000);
            
            return true;
        }
        
    },
    streamData: {
        data: [],
        trend: {data: [], q: [], qSize: 10},
        sums: {pCount: 0, nCount: 0, pSum: 0, nSum: 0 },
        getData: function() {
            var d = [
                {key: 'Tweets', color: 'green', values: SMM.streamData.data, type: 'line'}
            ];

            return d;
        },
        getTrendData: function() {
            return [
                {key: 'Polarity trend', color: 'gray', values: SMM.streamData.trend.data}
            ];
        },
        getSumData: function() {
            if(SMM.streamData.sums.pSum + Math.abs(SMM.streamData.sums.nSum) == 0){
                return [];
            }
            
            return [
                {
                    key: 'Positive ',
                    value: SMM.streamData.sums.pSum},
                {
                    key: 'Negative ',
                    value: Math.abs(SMM.streamData.sums.nSum)}
            ];
        },
        getCountData: function() {
            var t = SMM.streamData.sums.pCount + SMM.streamData.sums.nCount;
            if(!t){
                return [];
            }
            return [
                {
                    key: 'Positive (%)',
                    value: (SMM.streamData.sums.pCount / t)*100
                },
                {
                    key: 'Negative (%)',
                    value: (SMM.streamData.sums.nCount / t)*100
                }
            ];
        },
        push: function(data) {

            var d0 = {y: data.polarity, x: new Date(data.stamp), size: 0.2, text: data.text};

            if (data.polarity < 0) {
                d0.color = 'red';
                SMM.streamData.sums.nCount += 1;
                SMM.streamData.sums.nSum += d0.y;
                negativeHeatMapData.push(new google.maps.LatLng(data.original.geo.coordinates[0], data.original.geo.coordinates[1]));
            } else {
                SMM.streamData.sums.pCount += 1;
                SMM.streamData.sums.pSum += d0.y;
                positiveHeatMapData.push(new google.maps.LatLng(data.original.geo.coordinates[0], data.original.geo.coordinates[1]));
            }

            SMM.streamData.data.push(d0);

            var d1 = {y: data.polarity, x: new Date(data.stamp)};
            SMM.streamData.smooth(SMM.streamData.trend, d1);

        },
        smooth: function(container, d) {
            container.q.push(d);
            if (container.q.length > container.qSize) {
                var x = container.q[0].x;
                var y_sum = 0;

                for (var j = 0; j < container.qSize; j++) {
                    y_sum += container.q[0].y;
                    container.q.shift();
                }
                var c = 'green';
                if (y_sum / container.qSize < 0) {
                    c = 'red';
                }
                var a = {
                    y: y_sum / container.qSize,
                    x: x,
                    size: 2,
                    color: c
                };
                container.data.push(a);
            }
        },
        clear: function() {
            //SMM.streamData.data = [];
            //SMM.streamData.trend = { data:[], q : [], qSize: 10};
        }
    },
    charts: {
        chartPool: [],
        sumChartContainer: '#sum-chart svg',
        countChartContainer: '#count-chart svg',
        polartyChartContainer: '#polarity-chart svg',
        trendChartContainer: '#trend-chart svg',
        updateInt: 2000,
                
        init: function() {
            if(SMM.stream == null){
                return null;
            }
            
            nv.addGraph(function() {
                var chartSumColors = d3.scale.ordinal().range(['green', 'red']);

                var chartSum = nv.models.pieChart()
                        .showLegend(false).color(chartSumColors.range())
                        .x(function(d) {
                    return d.key 
                }).y(function(d) {
                    return d.value 
                });

                chartSum.updateManual = function() {
                    d3.select(SMM.charts.sumChartContainer).datum(SMM.streamData.getSumData).transition().duration(200).call(chartSum);
                };
                chartSum.updateManual();
                nv.utils.windowResize(chartSum.update);
                SMM.charts.chartPool.push(chartSum);
                
                var chartCount = nv.models.pieChart()
                        .showLegend(false).color(chartSumColors.range())
                        .x(function(d) {
                    return d.key 
                }).y(function(d) {
                    return d.value 
                });

                chartCount.updateManual = function() {
                    d3.select(SMM.charts.countChartContainer).datum(SMM.streamData.getCountData).transition().duration(200).call(chartCount);
                };
                chartCount.updateManual();
                nv.utils.windowResize(chartCount.update);
                SMM.charts.chartPool.push(chartCount);
                
                var chartDist = nv.models.scatterChart()
                        .showDistX(true)
                        .showDistY(true)
                        .showLegend(false);

                chartDist.xAxis.tickFormat(function(d) {
                    return d3.time.format("%H:%M:%S")(new Date(d))
                }).axisLabel('Time');
                chartDist.yAxis.tickFormat(d3.format('.02f')).axisLabel('Polarity');
                chartDist.tooltipContent(function(key, x, y, d) {
                    return "<div class='tooltipContainer'>" + d.point.text + "</div>";
                });

                d3.select(SMM.charts.polartyChartContainer).datum(SMM.streamData.getData).transition().duration(200).call(chartDist);
                nv.utils.windowResize(chartDist.update);

                SMM.charts.chartPool.push(chartDist);

                var chartTrend = nv.models.lineChart();
                chartTrend.showLegend(false);
                chartTrend.xAxis.tickFormat(function(d) {
                    return d3.time.format("%H:%M:%S")(new Date(d))
                }).axisLabel('Time');
                chartTrend.yAxis.tickFormat(d3.format('.02f')).axisLabel('Polarity');

                d3.select(SMM.charts.trendChartContainer).datum(SMM.streamData.getTrendData).transition().duration(200).call(chartTrend);
                nv.utils.windowResize(chartTrend.update);

                SMM.charts.chartPool.push(chartTrend);
            });

            setInterval(function() {
                SMM.charts.redraw()
            }, SMM.charts.updateInt);

        },
        redraw: function() {
            for (var i in SMM.charts.chartPool) {
                if (typeof SMM.charts.chartPool[i].updateManual === 'function') {
                    SMM.charts.chartPool[i].updateManual();
                }
            }

            var positiveHeatmap = new google.maps.visualization.HeatmapLayer({
              data: positiveHeatMapData
            });
            positiveHeatmap.setOptions({radius: 10, gradient: positiveGradient});
            positiveHeatmap.setMap(map);

            var negativeHeatmap = new google.maps.visualization.HeatmapLayer({
              data: negativeHeatMapData
            });
            negativeHeatmap.setOptions({radius: 10, gradient: negativeGradient});
            negativeHeatmap.setMap(map);
        }
    }
};

