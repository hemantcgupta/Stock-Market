import React, { Component } from 'react';
import ".././POSPromotionTracking/Allgraphs/allgraphs.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';


ReactFC.fcRoot(FusionCharts, Charts);


class NationalityGraph extends Component {
    constructor(props) {
        super(props);
        this.state = {
            NationalityChartConfig: {
                type: 'bar2d',
                renderAt: 'chart-container',
                width: '590',
                height: '320',
                dataFormat: 'json',
                dataSource: {
                    chart: {
                        "caption": "Revenue by PCSSR",
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
        let NationalityChartConfig = this.state.NationalityChartConfig;
        NationalityChartConfig.dataSource.data = props.datasetOne
        this.setState({ NationalityChartConfig: NationalityChartConfig })
    }

    componentWillReceiveProps = (props) => {
        let NationalityChartConfig = this.state.NationalityChartConfig;
        NationalityChartConfig.dataSource.data = props.datasetOne
        this.setState({ NationalityChartConfig: NationalityChartConfig })
    }



    render() {
        return (
            <div>
                {this.state.NationalityChartConfig.dataSource.data.length > 0 ? <ReactFC {...this.state.NationalityChartConfig} />
                    : 'No Data To Show'}
                {this.state.NationalityChartConfig.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>
        );
    }
}

export default NationalityGraph;
