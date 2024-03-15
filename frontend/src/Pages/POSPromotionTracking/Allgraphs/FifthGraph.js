import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';
import './allgraphs.scss'

ReactFC.fcRoot(FusionCharts, Charts);


class FifthGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            visitChartConfig: {
                type: 'line',
                renderAt: 'chart-container',
                width: '295',
                height: '250',
                dataFormat: 'json',
                dataSource: {
                    chart: {
                        "theme": "fusion",
                        "caption": "Rev by Issue Date",
                        "rotateLabels": "0",
                        // "maxLabelWidthPercent": 80,
                        //"subCaption": "Last week",
                        "xAxisName": "Issue Date",
                        //"yAxisName": "No. of Visitors",
                        "lineThickness": "2"
                    },
                    data: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let visitChartConfig = this.state.visitChartConfig;
        visitChartConfig.dataSource.data = props.datasetFifth
        this.setState({ visitChartConfig: visitChartConfig })
    }

    componentWillReceiveProps = (props) => {
        let visitChartConfig = this.state.visitChartConfig;
        visitChartConfig.dataSource.data = props.datasetFifth
        this.setState({ visitChartConfig: visitChartConfig })
    }


    render() {
        return (
            <div>
                {this.state.visitChartConfig.dataSource.data.length > 0 ? <ReactFC {...this.state.visitChartConfig} />
                    : 'No Data To Show'}
                {this.state.visitChartConfig.dataSource.data.length > 0 ? <div className='hide-watermarkWhiite'></div> : ''}
            </div>
        );
    }
}

export default FifthGraphConfig;
