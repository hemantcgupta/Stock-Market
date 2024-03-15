import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';
import './allgraphs.scss'


ReactFC.fcRoot(FusionCharts, Charts);

class FirstGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            barChartConfigs: {
                type: "bar2d", // The chart type
                width: "295", // Width of the chart
                height: "250", // Height of the chart
                dataFormat: "json", // Data type
                dataSource: {
                    chart: {
                        caption: "Promo Performance Revenue",
                        captionFontSize: "14",
                        captionAlignment: "left",
                        captionHorizontalPadding: "-125",
                        // subCaption: "In MMbbl = One Million barrels",
                        // xAxisName: "Country",
                        // yAxisName: "Reserves (MMbbl)",
                        numberSuffix: "",
                        theme: "candy"
                    },
                    data: []
                }
            }
        }
    }

    componentDidMount() {
        let props = this.props;
        let barChartConfigs = this.state.barChartConfigs;
        barChartConfigs.dataSource.data = props.datasetOne
        this.setState({ barChartConfigs: barChartConfigs })

    }

    componentWillReceiveProps = (props) => {
        let barChartConfigs = this.state.barChartConfigs;
        barChartConfigs.dataSource.data = props.datasetOne
        this.setState({ barChartConfigs: barChartConfigs })
    }


    render() {
        console.log(this.state.barChartConfigs, 'divyans')
        return (
            <div>
                {this.state.barChartConfigs.dataSource.data.length > 0 ? <ReactFC {...this.state.barChartConfigs} />
                    : 'No Data To Show'}
                {this.state.barChartConfigs.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>

        );
    }
}
export default FirstGraphConfig;
