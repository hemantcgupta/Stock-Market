import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
import * as d3 from "d3";
import Chart from '../../../../Component/DrawBarChart';
import Spinners from "../../../../spinneranimation";
import '../PosDashboard.scss';


const apiServices = new APIServices();

class TopFiveODs extends Component {
    constructor(props) {
        super();
        this.state = {
            topfiveODsData: [],
            agentsName: [],
            colors: ["#0571b0", "#92c5de", '#46b6ed', "#4CAF50", "#d5d5d5", "#92c5de", "#0571b0"],
            headerName: '',
            isLoading: false,
        };
    }

    componentWillReceiveProps = (props) => {
        this.getBarchartData(props);
    }

    getBarchartData = (props) => {
        const self = this;
        const { startDate, endDate, regionId, countryId, cityId } = props;
        if (self.state.headerName === 'TOP 10 AGENTS') {
            this.setState({ isLoading: true, topfiveODsData: [], agentsName: [] })
            apiServices.getAgentsBarChart(startDate, endDate, regionId, countryId, cityId).then((topAgentsBarChart) => {
                this.setState({ isLoading: false })
                if (topAgentsBarChart) {
                    if (topAgentsBarChart.TableData.length > 0) {
                        const graphData = topAgentsBarChart.TableData;
                        const agentsName = topAgentsBarChart.AgentNames;
                        this.setState(
                            { topfiveODsData: this.normalizeData(graphData), agentsName: agentsName },
                            () => this.drawBarChart(this.state.topfiveODsData, this.state.colors, 'topODAgentB', 'topODAgent')
                        );
                    }
                }
            });
        } else {
            this.setState({ isLoading: true })
            apiServices.getODsBarChart(startDate, endDate, regionId, countryId, cityId).then((top5ODsBarChart) => {
                this.setState({ isLoading: false, topfiveODsData: [], agentsName: [] })
                if (top5ODsBarChart) {
                    if (top5ODsBarChart.length > 0) {
                        this.setState(
                            { topfiveODsData: this.normalizeData(top5ODsBarChart) },
                            () => this.drawBarChart(this.state.topfiveODsData, this.state.colors, 'topODAgentB', 'topODAgent')
                        );
                    }
                }
            });
        }
    }

    normalizeData = (data) => {
        let normalizedData = [];
        data.forEach((d) => {
            let categoryObj = { category: '', values: [] };
            const keys = Object.keys(d);
            keys.forEach((k, i) => {
                if (i) {
                    categoryObj.values.push({ rate: k, value: d[k] });
                } else {
                    categoryObj.category = d[k];
                }
            })
            normalizedData.push(categoryObj);
        });
        return normalizedData;
    }

    drawBarChart = (data, colors, svgID, graphID) => {
        var margin = {
            top: 10,
            right: 30,
            bottom: 23,
            left: 85
        },
            width = 630 - margin.left - margin.right,
            height = 260 - margin.top - margin.bottom;

        // var tip = d3Tip().attr('class', 'd3-tip').offset([-10, 0]);

        var x0 = d3.scale.ordinal()
            .rangeRoundBands([0, width], .1);

        var x1 = d3.scale.ordinal();

        var y = d3.scale.linear()
            .range([height, 0]);

        var xAxis = d3.svg.axis()
            .scale(x0)
            .tickSize(0)
            .orient("bottom");

        var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left");

        var color = d3.scale.ordinal()
            .range(colors);

        d3.select(`#${svgID}`).remove();

        var svg = d3.select(`#${graphID}`).append("svg")
            .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
            .attr('id', `${svgID}`)
            .attr("preserveAspectRatio", "none")
            .attr("xmlns", "http://www.w3.org/2000/svg")
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // svg.call(tip)

        var categorysNames = data.map(function (d) { return d.category; });
        var rateNames = data[0].values.map(function (d) { return d.rate; });

        x0.domain(categorysNames);
        x1.domain(rateNames).rangeRoundBands([0, x0.rangeBand()]);
        y.domain([0, d3.max(data, function (category) { return d3.max(category.values, function (d) { return d.value; }); })]);

        var xaxis = svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis)
            .style('cursor', 'pointer')
            .style('width', '100px')
            .style('text-overflow', 'ellipsis')
            .style('white-space', 'nowrap')
            .style('overflow', 'hidden')

        var div = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        xaxis.selectAll(".tick")[0].forEach(function (d1) {
            var data = d3.select(d1).data();//get the data asociated with y axis
            d3.select(d1).on("mouseover", function (d) {
                //on mouse hover show the tooltip
                div.transition()
                    .duration(200)
                    .style("opacity", .9);
                div.html(data)
                    .style("left", (d3.event.pageX) + "px")
                    .style("top", (d3.event.pageY - 28) + "px")
                    .style('background', 'rgb(42, 63, 84)')
                    .style('padding', '5px');
            })
                .on("mouseout", function (d) {
                    //on mouse out hide the tooltip
                    div.transition()
                        .duration(500)
                        .style("opacity", 0);
                });
        })

        var wrap = function () {
            var self = d3.select(this),
                textLength = self.node().getComputedTextLength(),
                text = self.text();
            while (textLength > (50) && text.length > 0) {
                text = text.slice(0, -1);
                self.text(text + '...');
                textLength = self.node().getComputedTextLength();
            }
        };

        xaxis.selectAll('text')
            .style('text-anchor', 'center')
            .attr('fill', '#8a9299')
            .each(wrap);

        svg.append("g")
            .attr("class", "y axis")
            .style('opacity', '0')
            .call(yAxis)
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .style('font-weight', 'bold')



        svg.select('.y').transition().duration(0).delay(0).style('opacity', '1');
        // svg.select('.y').transition().duration(500).delay(1300).style('opacity', '1');

        //Adding Tool tip
        var divToolTip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style('color', 'white')
            .style('background', 'rgb(42, 63, 84)')
            .style('padding', '5px')
            .style("opacity", 1e-6);

        var slice = svg.selectAll(".slice")
            .data(data)
            .enter().append("g")
            .attr("class", "g")
            .attr("transform", function (d) { return "translate(" + x0(d.category) + ",0)"; });

        slice.selectAll("rect")
            .data(function (d) { return d.values; })
            .enter().append("rect")
            .attr("width", x1.rangeBand())
            .attr("x", function (d) { return x1(d.rate); })
            .style("fill", function (d) { return color(d.rate) })
            .attr("y", function (d) { return y(0); })
            .attr("height", function (d) { return height - y(0); })
            .on("mouseover", function (d) {
                d3.select(this).style("fill", d3.rgb(color(d.rate)).darker(2));
                divToolTip.transition()
                    .attr('class', "padding tooltip")
                    .duration(500)
                    .style("opacity", 1);
            })
            .on("mousemove", function (d) {
                divToolTip
                    .text(d.rate + " : " + window.numberFormat(d.value, 2)) // here we using numberFormat function from Dashboard-indicator.js
                    .style("left", (d3.event.pageX - 140) + "px")
                    .style("top", (d3.event.pageY - 10) + "px");
            })
            .on("mouseout", function (d) {
                d3.select(this).style("fill", color(d.rate));
                divToolTip.transition()
                    .duration(500)
                    .style("padding", "0px !important")
                    .style("opacity", 1e-6);
            })

        slice.selectAll("rect")
            .transition() 
            // .delay(function (d) { return Math.random() * 1000; })
            // .duration(1000)
            .delay(0)
            .duration(0)
            .attr("y", function (d) { return y(d.value); })
            .attr("height", function (d) { return height - y(d.value); });


        //Legend
        // var legend = svg.selectAll(".legend")
        //     .data(data[0].values.map(function (d) { return d.rate; }).reverse())
        //     .enter().append("g")
        //     .attr("class", "legend")
        //     .attr("transform", function (d, i) { return "translate(0," + i * 20 + ")"; })
        //     .style("opacity", "0");

        // legend.append("rect")
        //     .attr("x", width - 18)
        //     .attr("width", 18)
        //     .attr("height", 18)
        //     .style("fill", function (d) { return color(d); });

        // legend.append("text")
        //     .attr("x", width - 24)
        //     .attr("y", 9)
        //     .attr("dy", ".35em")
        //     .style("text-anchor", "end")
        //     .text(function (d) { return d; });

        // legend.transition().duration(500).delay(function (d, i) { return 1300 + 100 * i; }).style("opacity", "1");

    }

    headerChange(e) {
        this.setState({ headerName: e.target.value }, () => this.getBarchartData(this.props))
    }

    render() {
        const { colors, isLoading, topfiveODsData, agentsName } = this.state;
        return (
            <div className="x_panel tile" style={{ paddingTop: '0px', overflow: 'visible' }}>
                <div className="x_title reduce-margin">
                    <select class="select header-dropdown responsive-size" onChange={(e) => this.headerChange(e)}>
                        <option>TOP 5 O&Ds</option>
                        <option>TOP 10 AGENTS</option>
                    </select>
                    <ul className="nav navbar-right panel_toolbox">
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MutliBarGraphLegends i={true} data={topfiveODsData} agentsName={agentsName} colors={colors} />
                        </div>
                        <li onClick={() => this.props.history.push("/topMarkets")}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ? <Spinners /> : topfiveODsData.length === 0 ?
                    <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                    <div className='centered-graph'>
                        <div id="topODAgent"></div>
                        <MutliBarGraphLegends data={topfiveODsData} agentsName={agentsName} colors={colors} />
                    </div>}
            </div>
        )
    }
}

export default TopFiveODs;
