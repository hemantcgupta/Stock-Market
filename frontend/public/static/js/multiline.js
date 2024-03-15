function drawMultiLineChart(Data, DivID, RevenueName) {

    var margin = {
            top: 10,
            right: 80,
            bottom: 30,
            left: 90
        },
        width = 630 - margin.left - margin.right,
        height = 260 - margin.top - margin.bottom;

    // var parseDate = d3.time.format("%d-%b");
    // var parseDate = d3.time.format("%Y-%m-%d");

    var x = d3.scale.ordinal()
        .rangeRoundBands([0, width]);

    var y = d3.scale.linear()
        .range([height, 0]);

    var colors = ['#69c242', '#64bbe3', '#ffcc00', '#ff7300', '#cf2030'];

    // Do not include a domain
    var color = d3.scale.ordinal()
      .range(colors);
    // var color = d3.scale.category10();

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .tickFormat(function (d){
            monthname = GetMonthName(d);
            return monthname.substring(0, 3);
        });

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(6)
        .tickFormat(d3.format("s"));
        // .tickFormat(function (d){
        //       return numberFormat(d, 0);
        // });

    // xData gives an array of distinct 'Weeks' for which trends chart is going to be made.
    var xData = Data[0].values.map(function(d) {
        return d.month
    });

    var line = d3.svg.line()
        // .interpolate("basis")
        .x(function(d) { return x(d.month) + x.rangeBand() / 2; })
        .y(function(d) { return y(d.revenue); });

    var svg = d3.select("#" + DivID).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        //Colors
    color.domain(Data.map(function(d) {
        return d.key;
    }));

    x.domain(xData);

    var valueMax = d3.max(Data, function(r) {
        return d3.max(r.values, function(d) {
            return d.revenue;
        })
    });
    var valueMin = d3.min(Data, function(r) {
        return d3.min(r.values, function(d) {
            return d.revenue;
        })
    });

    y.domain([valueMin, valueMax]);

    //Drawing X Axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    // Drawing Horizontal grid lines.
    svg.append("g")
        .attr("class", "GridX")
        .selectAll("line.grid").data(y.ticks()).enter()
        .append("line")
        .attr({
            "class": "grid",
            "x1": x(xData[0]),
            "x2": x(xData[xData.length - 1]) + x.rangeBand() / 2,
            "y1": function(d) {
                return y(d);
            },
            "y2": function(d) {
                return y(d);
            }
        });

    // Drawing Y Axis
    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dx", "-4.5em")
        .attr("dy", "-5.5em")
        .style("text-anchor", "end")
        .text(RevenueName);

    // Drawing Lines for each segments
    var segment = svg.selectAll(".segment")
        .data(Data)
        .enter().append("g")
        .attr("class", "segment");

    segment.append("path")
        .attr("id", function(d) {
            return d.key;  })
        .attr("visible", 1)
            .attr("d", function(d) {
            return line(d.values); })
            .style("stroke", function(d) { 
            if(d.key == "2018"){
                colorChange();
            }else {
                return color(d.key);
            }
            })
            // .attr("stroke", "green")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr('class', 'line');
       
   

    function colorChange(){
        segment.append("path")      
        .attr("d", function(d) {
            return line(d.values.filter(function(d){
                return d.DataType == "Actuals" && d.key == "2018";
            })) 
        })
        .style("stroke", "#64bbe3")
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr('class', 'line');

       segment.append("path")
         .attr("d", function(d) {
             return line(d.values.filter(function(d){
                return d.DataType == "Forecast";})) 
        })
          .attr("stroke", "#eed202")
            .attr("stroke-width", 2)
            .attr("fill", "none");
    }

    // Creating Dots on line
    segment.selectAll("dot")
        .data(function(d) {
            return d.values;
        })
        .enter().append("circle")
        .attr("r", 5)
        .attr("cx", function(d) {
            return x(d.month) + x.rangeBand() / 2;
        })
        .attr("cy", function(d) {
            return y(d.revenue);
        })
        .style("stroke", "white")
        .style("fill", function(d) {
            if(d.DataType == "Actuals"){
                return color(this.parentNode.__data__.key);
            }else{
                return "#eed202";                
            }
        })
        .on("mouseover", mouseover)
        // .on('mouseover', tip.show)
        .on("mousemove", function(d) {
            divToolTip
                .text(this.parentNode.__data__.key + " : " + numberFormat(d.revenue, 2)) // here we using numberFormat function from Dashboard-indicator.js
                .style("left", (d3.event.pageX + 15) + "px")
                .style("top", (d3.event.pageY - 10) + "px");
        })
        .on("mouseout", mouseout);        
        // .on('mouseout', tip.hide)

    segment.append("text")
        .datum(function(d) {
            return {
                name: d.key,
                RevData: d.values[d.values.length - 1]
            };
        })
        .attr("transform", function(d) {
            var xpos = x(d.RevData.month) + x.rangeBand() / 2;
            return "translate(" + xpos + "," + y(d.RevData.revenue) + ")";
        })
        .attr("x", 8)
        .attr("dy", ".38em")
        .attr("class", "segmentText")
        .style('font-size','14px')
        .style('font-weight', 'bold')
        .attr("Segid", function(d) {
            return d.name;
        })
        .text(function(d) {
            return d.name;
        });

    d3.selectAll(".segmentText").on("click", function(d) {
        var tempId = d3.select(this).attr("Segid");
        var flgVisible = d3.select("#" + tempId).attr("visible");

        var newOpacity = flgVisible == 1 ? 0 : 1;
        flgVisible = flgVisible == 1 ? 0 : 1;

        // Hide or show the elements
        d3.select("#" + tempId).style("opacity", newOpacity)
            .attr("visible", flgVisible);

    });
    // Adding Tooltip
    var divToolTip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 1e-6);

    function mouseover() {
        divToolTip.transition()
            .duration(500)
            .style("opacity", 1);
    }

    function mouseout() {
        divToolTip.transition()
            .duration(500)
            .style("opacity", 1e-6);
    }

}

//Avgfare Multiline chart start
function drawAvgFareMultiLineChart(Data, DivID, AvgfareName) {

    var margin = {
            top: 10,
            right: 80,
            bottom: 30,
            left: 80
        },
        width = 630 - margin.left - margin.right,
        height = 260 - margin.top - margin.bottom;

    // var parseDate = d3.time.format("%d-%b");
    // var parseDate = d3.time.format("%Y-%m-%d");

    var x = d3.scale.ordinal()
        .rangeRoundBands([0, width]);

    var y = d3.scale.linear()
        .range([height, 0]);

    // var color = d3.scale.category10();

    var colors = ['#69c242', '#64bbe3', '#ffcc00', '#ff7300', '#cf2030'];
    // Do not include a domain
    var color = d3.scale.ordinal()
      .range(colors);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .tickFormat(function (d){
            monthname = GetMonthName(d);
            return monthname.substring(0, 3);
        });

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(6)
        .tickFormat(d3.format("s"));
        // .tickFormat(function (d){
        //       return numberFormat(d, 0);
        // });

    // xData gives an array of distinct 'Weeks' for which trends chart is going to be made.
    var xData = Data[0].values.map(function(d) {
        return d.month
    });

    var line = d3.svg.line()
        // .interpolate("basis")
        .x(function(d) { return x(d.month) + x.rangeBand() / 2; })
        .y(function(d) { return y(d.avgfare); });

    var svg = d3.select("#" + DivID).append("svg")
        // .attr("width", width + margin.left + margin.right)
        // .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        //Colors
    color.domain(Data.map(function(d) {
        return d.key;
    }));

    x.domain(xData);

    var valueMax = d3.max(Data, function(r) {
        return d3.max(r.values, function(d) {
            return d.avgfare;
        })
    });
    var valueMin = d3.min(Data, function(r) {
        return d3.min(r.values, function(d) {
            return d.avgfare;
        })
    });

    y.domain([valueMin, valueMax]);

    //Drawing X Axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    // Drawing Horizontal grid lines.
    svg.append("g")
        .attr("class", "GridX")
        .selectAll("line.grid").data(y.ticks()).enter()
        .append("line")
        .attr({
            "class": "grid",
            "x1": x(xData[0]),
            "x2": x(xData[xData.length - 1]) + x.rangeBand() / 2,
            "y1": function(d) {
                return y(d);
            },
            "y2": function(d) {
                return y(d);
            }
        });

    // Drawing Y Axis
    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dx", "-4.5em")
        .attr("dy", "-5em")
        .style("text-anchor", "end")
        .text(AvgfareName);

    // Drawing Lines for each segments
    var segment = svg.selectAll(".segment")
        .data(Data)
        .enter().append("g")
        .attr("class", "segment");

    segment.append("path")
        .attr("id", function(d) {
                return d.key;  })
            .attr("visible", 1)
                .attr("d", function(d) {
                return line(d.values); })
                .style("stroke", function(d) { 
                if(d.key == "2018"){
                    colorChange();
                }else {
                    return color(d.key);
                }
                })
                // .attr("stroke", "green")
                .attr("stroke-width", 2)
                .attr("fill", "none")
                .attr('class', 'line');
           
       

        function colorChange(){
            segment.append("path")      
            .attr("d", function(d) {
                return line(d.values.filter(function(d){
                    return d.DataType == "Actuals" && d.key == "2018";
                })) 
        })
            .style("stroke", "#64bbe3")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr('class', 'line');

           segment.append("path")
             .attr("d", function(d) {
                 return line(d.values.filter(function(d){
                    return d.DataType == "Forecast";})) 
            })
              .attr("stroke", "#eed202")
                .attr("stroke-width", 2)
                .attr("fill", "none");
        }

    // Creating Dots on line
    segment.selectAll("dot")
        .data(function(d) {
            return d.values;
        })
        .enter().append("circle")
        .attr("r", 5)
        .attr("cx", function(d) {
            return x(d.month) + x.rangeBand() / 2;
        })
        .attr("cy", function(d) {
            return y(d.avgfare);
        })
        .style("stroke", "white")
        .style("fill", function(d) {
            if(d.DataType == "Actuals"){
            return color(this.parentNode.__data__.key);
            }else{
                return "#eed202";                
            }
        })
        .on("mouseover", mouseover)
        // .on('mouseover', tip.show)
        .on("mousemove", function(d) {
            divToolTip
                .text(this.parentNode.__data__.key + " : " + numberFormat(d.avgfare, 2)) // here we using numberFormat function from Dashboard-indicator.js
                .style("left", (d3.event.pageX + 15) + "px")
                .style("top", (d3.event.pageY - 10) + "px");
        })
        .on("mouseout", mouseout);        
        // .on('mouseout', tip.hide)

    segment.append("text")
        .datum(function(d) {
            return {
                avgname: d.key,
                avgfareData: d.values[d.values.length - 1]
            };
        })
        .attr("transform", function(d) {
            var xpos = x(d.avgfareData.month) + x.rangeBand() / 2;
            return "translate(" + xpos + "," + y(d.avgfareData.avgfare) + ")";
        })
        .attr("x", 8)
        .attr("dy", ".38em")
        .attr("class", "segmentText")
        .style('font-size','14px')
        .style('font-weight', 'bold')
        .attr("Segid", function(d) {
            return d.avgname;
        })
        .text(function(d) {
            return d.avgname;
        });

    d3.selectAll(".segmentText").on("click", function(d) {
        var tempId = d3.select(this).attr("Segid");
        var flgVisible = d3.select("#" + tempId).attr("visible");

        var newOpacity = flgVisible == 1 ? 0 : 1;
        flgVisible = flgVisible == 1 ? 0 : 1;

        // Hide or show the elements
        d3.select("#" + tempId).style("opacity", newOpacity)
            .attr("visible", flgVisible);

    });
    // Adding Tooltip
    var divToolTip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 1e-6);

    function mouseover() {
        divToolTip.transition()
            .duration(500)
            .style("opacity", 1);
    }

    function mouseout() {
        divToolTip.transition()
            .duration(500)
            .style("opacity", 1e-6);
    }

}