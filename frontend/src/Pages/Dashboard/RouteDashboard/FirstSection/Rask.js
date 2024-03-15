import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MultiLineChartLegends from '../../../../Component/MultiLineChartLegends';
import Chart from '../../../../Component/DrawLineChart';
import String from '../../../../Constants/validator';
import Data from '../../../../Constants/validator';
import Spinners from "../../../../spinneranimation";
import '../RouteDashboard.scss';
import { format } from "date-fns";


const apiServices = new APIServices();

class Rask extends Component {
    constructor(props) {
        super();
        this.state = {
            raskData: [],
            colors: ['#c907ef', "#e0b7eb", "#f442bf", "#92c5de", "#0571b0", "#92c5de", "#0571b0"],
            isLoading: false,
            selectedTrend: 'Day',
            // selectedDelta: 'abs',
            selectedDelta: 'actual',
        };
    }

    componentWillReceiveProps = (props) => {
        this.getGraphData(props);
    }

    getGraphData(props) {
        const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
        const { selectedTrend, selectedDelta } = this.state;
        const group = String.removeQuotes(routeGroup)
        this.url = `/route?RouteGroup=${group}`;
        this.getRouteURL(group, regionId, countryId, routeId);
        this.setState({ isLoading: true, raskData: [] })

        let endDateArray = endDate.split('-')
        endDateArray.length = 2
        let a = [...endDateArray, ...["01"]]

        apiServices.getRaskBarChart(a.join('-'), endDate, routeGroup, regionId, countryId, routeId, selectedTrend, selectedDelta).then((raskData) => {
            this.setState({ isLoading: false })
            if (raskData) {
                if (raskData[0].length > 0) {
                    let graphData = [...raskData[0], ...raskData[1]]
                    if (graphData.length > 0) {
                        this.setState(
                            { raskData: Data.rectifiedDeltaData(graphData, selectedDelta) },
                            () => Chart.drawLineChart(this.state.raskData, this.state.colors, 'raskL', 'rask', selectedDelta));
                    }
                }
            }
        });
    }

    periodChange = (e) => {
        e.stopPropagation();
        this.setState({ selectedDelta: e.target.value }, () => { this.getGraphData(this.props) })
    }

    // barchart(data) {

    //     var margin = {
    //         top: 10,
    //         right: 30,
    //         bottom: 23,
    //         left: 85
    //     },
    //         width = 630 - margin.left - margin.right,
    //         height = 260 - margin.top - margin.bottom;

    //     // var tip = d3Tip().attr('class', 'd3-tip').offset([-10, 0]);

    //     var x0 = d3.scale.ordinal()
    //         .rangeRoundBands([0, width], .1);

    //     var x1 = d3.scale.ordinal();

    //     var y = d3.scale.linear()
    //         .range([height, 0]);

    //     var xAxis = d3.svg.axis()
    //         .scale(x0)
    //         .tickSize(0)
    //         .orient("bottom");

    //     var yAxis = d3.svg.axis()
    //         .scale(y)
    //         .orient("left");

    //     var color = d3.scale.ordinal()
    //         .range(this.state.colors);

    //     d3.select("#raskGraph").remove();

    //     var svg = d3.select("#rask").append("svg")
    //         .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
    //         .attr("preserveAspectRatio", "none")
    //         .attr("id", "raskGraph")
    //         .attr("xmlns", "http://www.w3.org/2000/svg")
    //         .append("g")
    //         .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    //     // svg.call(tip)

    //     var categorysNames = data.map(function (d) { return d.category; });
    //     var rateNames = data[0].values.map(function (d) { return d.rate; });

    //     x0.domain(categorysNames);
    //     x1.domain(rateNames).rangeRoundBands([0, x0.rangeBand()]);
    //     y.domain([0, d3.max(data, function (category) { return d3.max(category.values, function (d) { return d.value; }); })]);

    //     svg.append("g")
    //         .attr("class", "x axis")
    //         .attr("transform", "translate(0," + height + ")")
    //         .call(xAxis);

    //     svg.append("g")
    //         .attr("class", "y axis")
    //         .style('opacity', '0')
    //         .call(yAxis)
    //         .append("text")
    //         .attr("transform", "rotate(-90)")
    //         .attr("y", 6)
    //         .attr("dy", ".71em")
    //         .style("text-anchor", "end")
    //         .style('font-weight', 'bold')

    //     svg.select('.y').transition().duration(500).delay(1300).style('opacity', '1');

    //     //Adding Tool tip
    //     var divToolTip = d3.select("body").append("div")
    //         .attr("class", "tooltip")
    //         .style('color', 'white')
    //         .style('background', 'rgb(42, 63, 84)')
    //         .style('padding', '5px')
    //         .style("opacity", 1e-6);

    //     var slice = svg.selectAll(".slice")
    //         .data(data)
    //         .enter().append("g")
    //         .attr("class", "g")
    //         .attr("transform", function (d) { return "translate(" + x0(d.category) + ",0)"; });

    //     slice.selectAll("rect")
    //         .data(function (d) { return d.values; })
    //         .enter().append("rect")
    //         .attr("width", x1.rangeBand())
    //         .attr("x", function (d) { return x1(d.rate); })
    //         .style("fill", function (d) { return color(d.rate) })
    //         .attr("y", function (d) { return y(0); })
    //         .attr("height", function (d) { return height - y(0); })
    //         .on("mouseover", function (d) {
    //             d3.select(this).style("fill", d3.rgb(color(d.rate)).darker(2));
    //             divToolTip.transition()
    //                 .attr('class', "padding tooltip")
    //                 .duration(500)
    //                 .style("opacity", 1);
    //         })
    //         .on("mousemove", function (d) {
    //             divToolTip
    //                 .text(d.rate + " : " + window.numberFormat(d.value, 2)) // here we using numberFormat function from Dashboard-indicator.js
    //                 .style("left", (d3.event.pageX - 140) + "px")
    //                 .style("top", (d3.event.pageY - 10) + "px");
    //         })
    //         .on("mouseout", function (d) {
    //             d3.select(this).style("fill", color(d.rate));
    //             divToolTip.transition()
    //                 .duration(500)
    //                 .style("padding", "0px !important")
    //                 .style("opacity", 1e-6);
    //         })

    //     slice.selectAll("rect")
    //         .transition()
    //         .delay(function (d) { return Math.random() * 1000; })
    //         .duration(1000)
    //         .attr("y", function (d) { return y(d.value); })
    //         .attr("height", function (d) { return height - y(d.value); });


    //     //Legend
    //     // var legend = svg.selectAll(".legend")
    //     //     .data(data[0].values.map(function (d) { return d.rate; }).reverse())
    //     //     .enter().append("g")
    //     //     .attr("class", "legend")
    //     //     .attr("transform", function (d, i) { return "translate(0," + i * 20 + ")"; })
    //     //     .style("opacity", "0");

    //     // legend.append("rect")
    //     //     .attr("x", width - 18)
    //     //     .attr("width", 18)
    //     //     .attr("height", 18)
    //     //     .style("fill", function (d) { return color(d); });

    //     // legend.append("text")
    //     //     .attr("x", width - 24)
    //     //     .attr("y", 9)
    //     //     .attr("dy", ".35em")
    //     //     .style("text-anchor", "end")
    //     //     .text(function (d) { return d; });

    //     // legend.transition().duration(500).delay(function (d, i) { return 1300 + 100 * i; }).style("opacity", "1");


    // }

    getRouteURL(routeGroup, regionId, countryId, routeId) {
        let leg = window.localStorage.getItem('LegSelected')
        let flight = window.localStorage.getItem('FlightSelected')

        if (regionId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
        }
        if (countryId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
        }
        if (routeId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
        }
        if (leg !== null && leg !== 'Null' && leg !== '') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
        }
        if (flight !== null && flight !== 'Null' && flight !== '') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
        }
    }

    render() {
        const { colors, isLoading, raskData, selectedTrend } = this.state;
        const uniqueArr = [... new Set(raskData.map(data => data.Category))]

        return (
            <div className="x_panel tile" style={{ paddingBottom: '15px' }}>
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>RASK</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <label className='label'>Delta:</label>
                        <select class="select" onChange={(e) => this.periodChange(e)}>
                            <option value='actual'>Actual</option>
                            <option value='pick_up'>Pick-up</option>
                            <option value='pick_up_percent'>Pick-up(%)</option>
                            {/* <option value='abs'>Absolute</option>
                            <option value='percent'>Percentage</option> */}
                        </select>
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MultiLineChartLegends i={true} colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>
                        <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ?
                    <Spinners /> :
                    raskData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="rask"></div>
                            <MultiLineChartLegends colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>}
            </div>
        )

    }
}

export default Rask;
