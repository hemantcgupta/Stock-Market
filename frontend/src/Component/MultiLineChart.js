import React, { Component } from "react";
import APIServices from '../API/apiservices';
import Spinners from "../spinneranimation";
import Constant from '../Constants/validator';
import * as d3 from "d3";
import * as d3Tip from "d3-tip";
import { getYear } from "date-fns";


const apiServices = new APIServices();

class MultiLineChart extends Component {
    constructor(props) {
        super();
        this.state = {
            performanceData: [],
            isLoading: false,
            getYear: null,
            gettingMonth: null,
            colors: ['#f68803', '#cb3bf0', '#ffcc00'],
            showForecastGraph: false
        };
    }

    componentDidMount() {
        const { displayName, route, alert, forecast, gettingYear, gettingMonth, isDirectPOS } = this.props;
        let selectedData = this.props.selectedData
        this.setState({ isLoading: true })
        let getCabin = window.localStorage.getItem('CabinSelected')
        getCabin = getCabin !== null && getCabin !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(getCabin)) : "Null";


        if (route) {
            let group = window.localStorage.getItem('RouteGroupSelected')
            group = group !== null && group !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(group)) : "Network";
            let region = window.localStorage.getItem('RouteRegionSelected')
            region = region !== null && region !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(region)) : "*";
            let country = window.localStorage.getItem('RouteCountrySelected')
            country = country !== null && country !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(country)) : "*";
            let route = window.localStorage.getItem('RouteSelected')
            route = route !== null && route !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(route)) : "*";
            let leg = window.localStorage.getItem('LegSelected')
            leg = leg !== null && leg !== 'Null' ? `'${leg}'` : "*";
            let flight = window.localStorage.getItem('FlightSelected')
            flight = flight !== null && flight !== 'Null' ? `${flight}` : "*";

            if (forecast) {
                apiServices.getRouteLineChartsForecast(displayName, group, region, country, route, leg, flight, getCabin, gettingYear, gettingMonth).then((performanceData) => {
                    this.setState({ isLoading: false, showForecastGraph: true })
                    if (performanceData) {
                        if (performanceData.length > 0) {
                            let CY = performanceData[0]
                            let LY = performanceData[1]
                            let graphData = [...CY, ...LY]
                            if (performanceData[2]) {
                                let TGT = performanceData[2]
                                graphData = [...graphData, ...TGT]
                            }
                            this.setState(
                                { performanceData: this.normalizeData(graphData) },
                                () => this.lineChart(this.state.performanceData));
                        }
                    }
                });
            }

            else {
                apiServices.getRouteLineCharts(displayName, group, region, country, route, leg, flight, getCabin).then((performanceData) => {
                    this.setState({ isLoading: false, showForecastGraph: false })
                    if (performanceData) {
                        if (performanceData.length > 0) {
                            let CY = performanceData[0]
                            let LY = performanceData[1]
                            let graphData = [...CY, ...LY]
                            if (performanceData[2]) {
                                let TGT = performanceData[2]
                                graphData = [...graphData, ...TGT]
                            }
                            this.setState(
                                { performanceData: this.normalizeData(graphData) },
                                () => this.lineChart(this.state.performanceData));
                        }
                    }
                });
            }
        } else {
            let region = window.localStorage.getItem('RegionSelected')
            region = region !== null && region !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(region)) : "*";
            let country = window.localStorage.getItem('CountrySelected')
            country = country !== null && country !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(country)) : "*";
            let city = window.localStorage.getItem('CitySelected')
            city = city !== null && city !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(city)) : "*";
            let od = window.localStorage.getItem('ODSelected')
            od = od !== null && od !== 'Null' ? Constant.addQuotesforMultiSelect(JSON.parse(od)) : "*";

            if (forecast) {
                apiServices.getPOSLineChartsForecast(displayName, region, country, city, od, getCabin, gettingYear, gettingMonth).then((performanceData) => {
                    this.setState({ isLoading: false, showForecastGraph: true })
                    if (performanceData) {
                        if (performanceData.length > 0) {
                            let CY = performanceData[0]
                            let LY = performanceData[1]
                            let graphData = [...CY, ...LY]
                            if (performanceData[2]) {
                                let TGT = performanceData[2]
                                graphData = [...graphData, ...TGT]
                            }
                            this.setState(
                                { performanceData: this.normalizeData(graphData) },
                                () => this.lineChart(this.state.performanceData));
                        }
                    }
                });
            }
            else if (alert) {
                let request
                selectedData = Constant.addQuotesforMultiSelect(selectedData)
                if (selectedData && region === "*" && !isDirectPOS) {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, selectedData, country, city, od, getCabin, "Infare")
                } else if (selectedData && country === "*" && !isDirectPOS) {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, region, selectedData, city, od, getCabin, "Infare")
                } else if (selectedData && city === "*") {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, region, country, selectedData, od, getCabin, "Infare")
                } else if (selectedData && od === "*") {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, region, country, city, selectedData, getCabin, "Infare")
                } else if (selectedData && getCabin === "Null") {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, region, country, city, od, selectedData, "Infare")
                } else {
                    request = apiServices.getInfareTrendData(gettingYear, gettingMonth, region, country, city, od, getCabin, "Infare")
                }
                request.then((result) => {
                    this.setState({ isLoading: false, showForecastGraph: true })
                    const tableData = result.TableData
                    let graphData = []

                    if (tableData) {
                        if (tableData.length > 0) {
                            graphData = tableData.map(el => {
                                return {
                                    Category: "Fare",
                                    Month: el.Alert_Type,
                                    value: el.MH_Fare,
                                }
                            }
                            )
                            this.setState(
                                { performanceData: this.normalizeData(graphData) },
                                () => this.lineChart(this.state.performanceData, undefined, this.props.alert));
                        }
                    }
                })
            }
            else {
                apiServices.getPOSLineCharts(displayName, region, country, city, od, getCabin).then((performanceData) => {
                    this.setState({ isLoading: false, showForecastGraph: false })
                    if (performanceData) {
                        if (performanceData.length > 0) {
                            let CY = performanceData[0]
                            let LY = performanceData[1]
                            let graphData = [...CY, ...LY]
                            if (performanceData[2]) {
                                let TGT = performanceData[2]
                                graphData = [...graphData, ...TGT]
                            }
                            this.setState(
                                { performanceData: this.normalizeData(graphData) },
                                () => this.lineChart(this.state.performanceData));
                        }
                    }
                });
            }
        }
    }

    normalizeData(data) {
        var result = data.map(function (el) {
            var o = Object.assign({}, el);
            o.DataType = 'Actuals';
            return o;
        })
        return result
    }

    lineChart(data, forecast, alert) {
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
        // var color = d3.scale.category10();

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
            return d.Month;
        });

        var line = d3.svg.line()
            // .interpolate("basis")
            .x(function (d) {
                return x(d.Month) + x.rangeBand() / 2;
            })
            .y(function (d) {
                return y(d.value);
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
                return x(d.Month) + x.rangeBand() / 2;
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
                    .text((alert ? d.Month : this.parentNode.__data__.key) + " : " + window.numberFormat(d.value, 2)) // here we using numberFormat function from Dashboard-indicator.js
                    .style("left", (d3.event.pageX + 15) + "px")
                    .style("top", (d3.event.pageY - 10) + "px");
            })
            .on("mouseout", mouseout);


        segment.selectAll("dot")
            .data(function (d) {
                return d.values;
            })
            .enter().append("text")
            .attr("x", function (d) {
                return (x(d.Month) + x.rangeBand() / 2) - 12;
            })
            .attr("y", function (d) {
                return y(d.value) - 5;
            })
            .attr("class", function (d) {
                return this.parentNode.__data__.key + "-dots hidden-texts";
            })
            .style("fill", "#000000")
            .html(function (d) {
                return window.numberFormat(d.value, 2);
            });


        segment.append("text")
            .datum(function (d) {
                return {
                    name: d.key,
                    RevData: d.values[d.values.length - 1]
                };
            })
            .attr("transform", function (d) {
                var xpos = x(d.RevData.Month) + x.rangeBand() / 2;
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

    LegendMouseEnter(line, showHighlight) {
        if (showHighlight) {
            var parentNode = document.querySelector('.show-hoverable-legends');
            parentNode.classList.add('hovered');
            var selectedItems = document.querySelectorAll('.' + line + '-dots');
            selectedItems.forEach(function (each) {
                each.classList.add('show');
            });
            var lines = document.querySelectorAll('path.line');
            lines.forEach(function (each) {
                if (each.id === line)
                    each.style.strokeWidth = "2.5";
                else
                    each.style.opacity = "0.2";
            });
        }
    }

    LegendMouseLeave(line, showHighlight) {
        if (showHighlight) {
            var parentNode = document.querySelector('.show-hoverable-legends');
            parentNode.classList.remove('hovered');
            var selectedItems = document.querySelectorAll('.' + line + '-dots');
            selectedItems.forEach(function (each) {
                each.classList.remove('show');
            });
            var lines = document.querySelectorAll('path.line');
            lines.forEach(function (each) {
                each.style.opacity = "1";
                each.style.strokeWidth = "1.5";
            });
        }
    }

    render() {
        const { isLoading, performanceData, colors, showForecastGraph } = this.state;
        const uniqueArr = [... new Set(performanceData.map(data => data.Category))];
        const showHoverableLegends = showForecastGraph ? 'show-hoverable-legends' : '';


        if (isLoading) {
            return (
                <Spinners />)
        } else {
            return (
                performanceData.length === 0 ?
                    <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                    <div className={showHoverableLegends} style={{ height: '270px' }}>
                        <div id="lineChart" style={{ height: '100%' }}></div>
                        <div id="legend" className='col-md-12'>
                            {uniqueArr.map((d, i) =>
                                <p className={'sub-legend cursor-hover-legend ' + d + '-line'} onMouseEnter={this.LegendMouseEnter.bind(this, d, showForecastGraph)} onMouseLeave={this.LegendMouseLeave.bind(this, d, showForecastGraph)} ><div className="square" style={{ background: colors[i] }}></div>{d}</p>
                            )}
                        </div>
                    </div>
            )
        }
    }
}

export default MultiLineChart;
