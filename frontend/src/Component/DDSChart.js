import React from "react";
import ReactDOM from "react-dom";
import ReactFC from "react-fusioncharts";
import FusionCharts from "fusioncharts";
import Column2D from "fusioncharts/fusioncharts.charts";
import FusionTheme from "fusioncharts/themes/fusioncharts.theme.candy";

ReactFC.fcRoot(FusionCharts, Column2D, FusionTheme);


class DDSChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            chartConfigs: {
                type: "mscolumn2d",
                renderAt: "chart-container",
                width: "100%",
                height: "90%",
                dataFormat: "json",
                dataSource: {
                    chart: {
                        caption: "Year on Year Growth",
                        yaxisname: `Passenger`,
                        subcaption: "(by Travel Period)",
                        yaxismaxvalue: "100",
                        yaxisminvalue: "-100",
                        numberSuffix: '%',
                        showsum: "1",
                        plottooltext:
                            "$seriesName in $label : <b>$dataValue</b>",
                        decimals: "1",
                        theme: "candy",
                        animation:'0' 
                    },
                    categories: [],
                    dataset: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let chartConfigs = this.state.chartConfigs;
        chartConfigs.dataSource.chart.yaxisname = props.yaxisname
        chartConfigs.dataSource.categories = props.categories
        chartConfigs.dataSource.dataset = props.dataset
        this.setState({ chartConfigs: chartConfigs })
    }

    componentWillReceiveProps = (props) => {
        let chartConfigs = this.state.chartConfigs;
        chartConfigs.dataSource.chart.yaxisname = props.yaxisname
        chartConfigs.dataSource.categories = props.categories
        chartConfigs.dataSource.dataset = props.dataset
        this.setState({ chartConfigs: chartConfigs })
    }

    render() {
        return (<ReactFC {...this.state.chartConfigs} />);
    }
}

export default DDSChart;