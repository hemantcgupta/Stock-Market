import React, { Component } from 'react';
import ".././POSPromotionTracking/Allgraphs/allgraphs.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';

ReactFC.fcRoot(FusionCharts, Charts);


class AgeBandGraph extends Component {
    constructor(props) {
        super(props);
        this.state = {
            AgeBandChartConfig: {
                type: 'bar2d',
                renderAt: 'chart-container',
                width: '590',
                height: '320',
                dataFormat: 'json',
                dataSource: {
                    chart: {
                        "theme": "fusion",
                        "caption": " Revenue by PCSSR",
                        "rotateLabels": "0",
                        "labelDisplay": "notwrap",
                        "captionFontSize": "12",
                        //"subCaption": "Last week",
                        "xAxisName": "pcssr",
                        //"yAxisName": "No. of Visitors",
                        "lineThickness": "2",
                        "theme": 'candy'
                    },
                    data: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let AgeBandChartConfig = this.state.AgeBandChartConfig;
        AgeBandChartConfig.dataSource.data = props.datasetFifth
        this.setState({ AgeBandChartConfig: AgeBandChartConfig })
    }

    componentWillReceiveProps = (props) => {
        let AgeBandChartConfig = this.state.AgeBandChartConfig;
        AgeBandChartConfig.dataSource.data = props.datasetFifth
        this.setState({ AgeBandChartConfig: AgeBandChartConfig })
    }



    render() {
        return (
            <div>
                {this.state.AgeBandChartConfig.dataSource.data.length > 0 ? <ReactFC {...this.state.AgeBandChartConfig} />
                    : 'No Data To Show'}
                {this.state.AgeBandChartConfig.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>
        );
    }
}

export default AgeBandGraph;
