import React, { Component } from "react";
import APIServices from '../API/apiservices';
import Spinners from "../spinneranimation";
import * as d3 from "d3";
import * as d3Tip from "d3-tip";


const apiServices = new APIServices();

class InfareMultiLineChart extends Component {
    constructor(props) {
        super();
        this.state = {
            performanceData: [],
            isLoading: false,
            colors: ['#f68803', '#cb3bf0', '#ffcc00', "#009688", "#4CAF50", "#03f574", "#92c5de", "#0571b0", "#92c5de", "#0571b0"]
        };
    }


    normalizeData = (data) => {
        var result = data.map(function (el) {
            var o = Object.assign({}, el);
            o.DataType = 'Actuals';
            return o;
        })

        // let sortedData = result.sort((a, b) => a.AverageFare_CY - b.AverageFare_CY)
        return result
    }

    lineChart = (data) => {
        var Data = d3.nest()
            .key(function (d) {
                return d.Carrier;
                // return 'revenue';
            })
            .entries(data);
        var margin = {
            top: 10,
            right: 30,
            bottom: 15,
            left: 70
        },
            width = 610 - margin.left - margin.right,
            height = 260 - margin.top - margin.bottom;

        var x = d3.scale.ordinal()
            .rangeRoundBands([0, width]);

        var y = d3.scale.linear()
            .range([height, 0]);

        var colors = this.state.colors;

        // Do not include a domain
        var color = d3.scale.ordinal()
            .range(colors);
        // var color = d3.scale.Carrier10();

        var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom")
            .tickFormat(function (d) {
                return d;
            });

        var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left")
            .ticks(6)
            .tickFormat(d3.format("s"));

        var xData = Data[0].values.map(function (d) {
            return d.DepartureDate
        });

        var line = d3.svg.line()
            // .interpolate("basis")
            .x(function (d) {
                return x(d.DepartureDate) + x.rangeBand() / 2;
            })
            .y(function (d) {
                return y(d.AverageFare_CY);
            });

        d3.select("#rtg").remove();

        var svg = d3.select("#lineChart").append("svg")
            .attr("viewBox", "0 0 600 280")
            .attr('id', 'rtg')
            .attr('style', 'overflow:visible')
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
                return d.AverageFare_CY;
            })
        });
        var valueMin = d3.min(Data, function (r) {
            return d3.min(r.values, function (d) {
                return d.AverageFare_CY;
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
            .style("font-family", '"Helvetica Neue",Roboto,Arial,"Droid Sans",sans-serif')
            .style("font-size", "14px")
            .style("font-weight", "500")

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
                if (d.Carrier === "AK") {
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
                        return d.DataType === "Actuals" && d.Carrier === "AK";
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
                return x(d.DepartureDate) + x.rangeBand() / 2;
            })
            .attr("cy", function (d) {
                return y(d.AverageFare_CY);
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
                    .text(this.parentNode.__data__.key + " : " + window.numberFormat(d.AverageFare_CY, 2)) // here we using numberFormat function from Dashboard-indicator.js
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
                var xpos = x(d.RevData.DepartureDate) + x.rangeBand() / 2;
                return "translate(" + (xpos - 9) + "," + (y(d.RevData.AverageFare_CY) - 10) + ")";
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
            .style('background', 'rgb(42, 63, 84)')
            .style('padding', '5px')
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

        /* Optionally add a guideline */
        var guideline = svg.append('line')
            .attr('stroke', '#333')
            .attr('stroke-width', 0)
            .attr('class', 'guide')
            .attr('x1', 1)
            .attr('y1', 1)
            .attr('x2', 1)
            .attr('y2', height)

    }

    render() {
        let { colors } = this.state;
        let infareData = this.props.infareData;
        const uniqueArr = [... new Set(infareData.map(data => data.Carrier))]

        if (infareData.length > 0) {
            console.log('rahul', this.normalizeData(infareData))
            this.lineChart(this.normalizeData(infareData))
        }
        return (
            infareData.length === 0 ?
                <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                <div style={{ height: '270px' }}>
                    <div id="lineChart" style={{ height: '100%' }}>
                        <div className='y-axis'>{this.props.currency}</div>
                    </div>
                    <div className='x-axis'>Departure Date</div>
                    <div id="legend" className='col-md-12'>
                        {uniqueArr.map((d, i) =>
                            <p className="sub-legend"><div className="square" style={{ background: colors[i] }}></div>{`${d}`}</p>
                        )}
                    </div>
                </div>
        )
    }
}

export default InfareMultiLineChart;
