import React, { Component } from 'react';
import "../../.././Component/component.scss";
import FusionCharts from 'fusioncharts';
import Charts from 'fusioncharts/fusioncharts.charts';
import ReactFC from 'react-fusioncharts';
import './allgraphs.scss'


ReactFC.fcRoot(FusionCharts, Charts);

class ThirdGraphConfig extends Component {
    constructor(props) {
        super(props);
        this.state = {
            pieChartConfigs: {
                type: 'pie2d',
                renderAt: 'chart-container',
                width: '295',
                height: '300',
                dataFormat: 'json',
                dataSource: {
                    "chart": {
                        caption: "Top 10 Non Directional ODs - by Rev (MYR-Mil)",
                        captionFontSize: "14",
                        captionAlignment: "left",
                        // "subCaption": "Last year",
                        "numberPrefix": "$",
                        "showPercentInTooltip": "0",
                        "decimals": "1",
                        //Theme
                        "theme": "candy",
                        "tooltipBorderRadius": "20"
                    },
                    data: []
                }
            }
        }
    }

    componentWillMount() {
        let props = this.props;
        let pieChartConfigs = this.state.pieChartConfigs;
        pieChartConfigs.dataSource.data = props.datasetThree
        this.setState({ pieChartConfigs: pieChartConfigs })
    }

    componentWillReceiveProps = (props) => {
        let pieChartConfigs = this.state.pieChartConfigs;
        pieChartConfigs.dataSource.data = props.datasetThree
        this.setState({ pieChartConfigs: pieChartConfigs })
    }

    render() {
        return (
            <div>
                {this.state.pieChartConfigs.dataSource.data.length > 0 ? <ReactFC {...this.state.pieChartConfigs} />
                    : 'No Data To Show'}
                {this.state.pieChartConfigs.dataSource.data.length > 0 ? <div className='hide-watermark'></div> : ''}
            </div>
        );
    }
}
export default ThirdGraphConfig;
