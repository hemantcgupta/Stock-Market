import * as d3 from "d3";

const drawLineChart = (data, colors, svgID, graphID, selectedType) => {
    var Data = d3.nest()
        .key(function (d) {
            return d.Category;
            // return 'revenue';
        })
        .entries(data);
    var margin = {
        top: 10,
        right: 30,
        bottom: 15,
        left: 70
    },
        width = 620 - margin.left - margin.right,
        height = 260 - margin.top - margin.bottom;

    var x = d3.scale.ordinal()
        .rangeRoundBands([0, width]);

    var y = d3.scale.linear()
        .range([height, 0]);

    var colors = colors;

    // Do not include a domain
    var color = d3.scale.ordinal()
        .range(colors);
    // var color = d3.scale.category10();

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .tickFormat(function (d) {
            return d;
        });

    var disableFormating = false;
    data.map((d) => d.value < 1000 && d.value > 0 ? disableFormating = true : null)

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(6)
        .tickFormat(d3.format(`${disableFormating ? "" : "s"}`));

    var xData = Data[0].values.map(function (d) {
        return d.Xaxis
    });

    var line = d3.svg.line()
        // .interpolate("basis")
        .x(function (d) {
            return x(d.Xaxis) + x.rangeBand() / 2;
        })
        .y(function (d) {
            return y(d.value);
        });

    d3.select(`#${svgID}`).remove();

    var svg = d3.select(`#${graphID}`).append("svg")
        .attr("viewBox", "0 0 600 280")
        .attr('id', `${svgID}`)
        .attr('style', 'overflow:hidden; height: 100%; width:100%')
        .attr("preserveAspectRatio", "none")
        .attr("xmlns", "http://www.w3.org/2000/svg")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    //Colors
    color.domain(Data.map(function (d) {
        return d.key;
    }));

    x.domain(xData);
    var valueMax = d3.max(Data, function (r) {
        return d3.max(r.values, function (d) {
            return d.value;
        })
    });
    var valueMin = d3.min(Data, function (r) {
        return d3.min(r.values, function (d) {
            return d.value;
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
            "y1": function (d) {
                return y(d);
            },
            "y2": function (d) {
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
        .attr("dx", "-5em")
        .attr("dy", "-4em")
        .style("text-anchor", "end")
        .style("font-weight", "bold")

    // Drawing Lines for each segments
    var segment = svg.selectAll(".segment")
        .data(Data)
        .enter().append("g")
        .attr("class", "segment");

    segment.append("path")
        .attr("id", function (d) {
            return d.key;
        })
        .attr("visible", 1)
        .attr("d", function (d) {
            return line(d.values);
        })
        .style("stroke", function (d) {
            if (d.Category === "LY") {
                colorChange();
            } else {
                return color(d.key);
            }
        })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr('class', 'line');


    function colorChange() {
        segment.append("path")
            .attr("d", function (d) {
                return line(d.values.filter(function (d) {
                    return d.DataType === "Actuals" && d.Category === "LY";
                }))
            })
            .style("stroke", "#64bbe3")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr('class', 'line');

        segment.append("path")
            .attr("d", function (d) {
                return line(d.values.filter(function (d) {
                    return d.DataType === "Actuals";
                }))
            })
            .attr("stroke", "#eed202")
            .attr("stroke-width", 2)
            .attr("fill", "none");
    }

    // Creating Dots on line
    segment.selectAll("dot")
        .data(function (d) {
            return d.values;
        })
        .enter().append("circle")
        .attr("r", 5)
        .attr("cx", function (d) {
            return x(d.Xaxis) + x.rangeBand() / 2;
        })
        .attr("cy", function (d) {
            return y(d.value);
        })
        .style("stroke", "white")
        .style("cursor", "pointer")
        .style("fill", function (d) {
            if (d.DataType === "Actuals") {
                return color(this.parentNode.__data__.key);
            } else {
                return "#eed202";
            }
        })
        .on("mouseover", mouseover)
        .on("mousemove", function (d) {
            divToolTip
                // .text(this.parentNode.__data__.key + " : " + window.numberFormat(d.value, 2) + " , " + d.delta)
                .html(`<p class="tooltipKey">${this.parentNode.__data__.key}</p><div class="tooltipMain"> <p class="tooltipValue">${d.value > 0 ? `${window.positiveDeltaFormat(d.value)}${selectedType === 'pick_up_percent' ? '%' : ''} ` : `${window.negativeDeltaFormat(d.value)}${selectedType === 'pick_up_percent' ? '%' : ''}`}</p> <div class="delta ${selectedType === 'actual' ? 'delta_display' : ''}"><p class="tooltipDelta ${selectedType === 'actual' ? d.delta_absolute.includes('-') ? 'red' : '' : ''}">(${d.delta_absolute})</p><p class="tooltipDelta ${selectedType === 'actual' ? d.delta_percent.includes('-') ? 'red' : '' : ''}">(${d.delta_percent})</p></div></div>`)
                .style("left", (d3.event.pageX + 15) + "px")
                .style("top", (d3.event.pageY - 10) + "px");
        })
        .on("mouseout", mouseout);



    segment.append("text")
        .datum(function (d) {
            return {
                name: d.key,
                RevData: d.values[d.values.length - 1]
            };
        })
        .attr("transform", function (d) {
            var xpos = x(d.RevData.Xaxis) + x.rangeBand() / 2;
            return "translate(" + (xpos - 9) + "," + (y(d.RevData.value) - 10) + ")";
        })
        .attr("x", -5)
        .attr("y", -5)
        .attr("dy", ".38em")
        .attr("class", "segmentText")
        .style('font-size', '0px')
        .style('font-weight', 'bold')
        .attr("Segid", function (d) {
            return d.name;
        })
        .text(function (d) {
            return d.name;
        });

    // Adding Tooltip
    var divToolTip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style('color', 'white')
        .style('width', 'auto')
        .style('background', 'rgb(42, 63, 84)')
        .style('padding', '5px')
        .style('padding-bottom', '-5px')
        .style("opacity", 1e-6);

    function mouseover() {
        divToolTip.transition()
            .attr('class', "padding tooltip")
            .duration(500)
            .style("opacity", 1);
    }

    function mouseout() {
        divToolTip.transition()
            .duration(500)
            .style("padding", "0px !important")
            .style("opacity", 1e-6);
    }

    /* Add 'curtain' rectangle to hide entire graph */
    // var curtain = svg.append('rect')
    //     .attr('x', -1 * width)
    //     .attr('y', -1 * height)
    //     .attr('height', height + 15)
    //     .attr('width', width + 80)
    //     .attr('class', 'curtain')
    //     .attr('transform', 'rotate(180)')
    //     .style('fill', '#2e303f')

    /* Optionally add a guideline */
    var guideline = svg.append('line')
        .attr('stroke', '#333')
        .attr('stroke-width', 0)
        .attr('class', 'guide')
        .attr('x1', 1)
        .attr('y1', 1)
        .attr('x2', 1)
        .attr('y2', height)

    /* Create a shared transition for anything we're animating */
    // var t = svg.transition()
    //     .delay(750)
    //     .duration(2000)
    //     .ease('linear')
    //     .each('end', function () {
    //         d3.select('line.guide')
    //             .transition()
    //             .style('opacity', 0)
    //             .remove()
    //     });

    // t.select('rect.curtain')
    //     .attr('width', 0);
    // t.select('line.guide')
    //     .attr('transform', 'translate(' + (width + 100) + ', 0)')

    // d3.select("#show_guideline").on("change", function (e) {
    //     guideline.attr('stroke-width', this.checked ? 1 : 0);
    //     curtain.attr("opacity", this.checked ? 0.75 : 1);
    // })

}

export default { drawLineChart };