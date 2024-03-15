import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';

ReactFC.fcRoot(FusionCharts, Charts);


class FourthGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            doughnutChartConfig: {
                type: 'doughnut2d',
                renderAt: 'chart-container',
                width: '295',
                height: '300',
                dataFormat: 'json',
                dataSource: {
                    "chart": {
                        caption: "Channel Contribution (Rev)",
                        captionFontSize: "14",
                        captionAlignment: "left",
                        //"subCaption": "Last year",
                        "numberPrefix": "$",
                        "startingAngle": "310",
                        "defaultCenterLabel": "Total revenue: $64.08K",
                        "centerLabel": "Revenue from $label: $value",
                        "centerLabelBold": "1",
                        "showTooltip": "0",
                        "decimals": "0",
                        "theme": "candy"
                    },
                    data: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let doughnutChartConfig = this.state.doughnutChartConfig;
        doughnutChartConfig.dataSource.data = props.datasetFourth
        this.setState({ doughnutChartConfig: doughnutChartConfig })
    }

    componentWillReceiveProps = (props) => {
        let doughnutChartConfig = this.state.doughnutChartConfig;
        doughnutChartConfig.dataSource.data = props.datasetFourth
        this.setState({ doughnutChartConfig: doughnutChartConfig })
    }

    render() {
        return (
            <div>
                {this.state.doughnutChartConfig.dataSource.data.length > 0 ? <ReactFC {...this.state.doughnutChartConfig} />
                    : 'No Data To Show'}
                {this.state.doughnutChartConfig.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>
        );
    }
}
export default FourthGraphConfig;
