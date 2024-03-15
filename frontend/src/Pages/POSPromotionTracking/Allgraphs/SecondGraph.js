import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';
import './allgraphs.scss'


ReactFC.fcRoot(FusionCharts, Charts);

class SecondGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            heatChartConfigs: {
                type: "bar2d", // The chart type
                width: "295", // Width of the chart
                height: "250", // Height of the chart
                dataFormat: "json", // Data type
                dataSource: {
                    chart: {
                        caption: "No of Tickets by Channel",
                        captionFontSize: "14",
                        captionAlignment: "left",
                        captionHorizontalPadding: "-89",
                        //subCaption: "No. of Tickets by Channel",
                        // subCaptionbgColor: 'white',
                        // // xAxisName: "Country",
                        // yAxisName: "Reserves (MMbbl)",
                        numberSuffix: "K",
                        theme: "candy"
                    },
                    data: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let heatChartConfigs = this.state.heatChartConfigs;
        heatChartConfigs.dataSource.data = props.datasetTwo
        this.setState({ heatChartConfigs: heatChartConfigs })
    }

    componentWillReceiveProps = (props) => {
        let heatChartConfigs = this.state.heatChartConfigs;
        heatChartConfigs.dataSource.data = props.datasetTwo
        this.setState({ heatChartConfigs: heatChartConfigs })
    }

    render() {
        return (
            <div>
                {this.state.heatChartConfigs.dataSource.data.length > 0 ? <ReactFC {...this.state.heatChartConfigs} />
                    : 'No Data To Show'}
                {this.state.heatChartConfigs.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>
        );
    }
}
export default SecondGraphConfig;
