import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';

ReactFC.fcRoot(FusionCharts, Charts);


class EightthGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            visitChartConfig: {
                type: 'line',
                renderAt: 'chart-container',
                width: '295',
                height: '250.5',
                dataFormat: 'json',
                dataSource: {
                    chart: {
                        "theme": "fusion",
                        "caption": "No of Tickets by Travel Period",
                        "rotateLabels": "0",
                        "labelDisplay": "notwrap",
                        "captionFontSize": "12",
                        //"subCaption": "Last week",
                        "xAxisName": "Travel Period",
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
        visitChartConfig.dataSource.data = props.datasetEightth
        this.setState({ visitChartConfig: visitChartConfig })
    }

    componentWillReceiveProps = (props) => {
        let visitChartConfig = this.state.visitChartConfig;
        visitChartConfig.dataSource.data = props.datasetEightth
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

export default EightthGraphConfig;
