import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
import Chart from '../../../../Component/DrawBarChart';
import Spinners from "../../../../spinneranimation";

const apiServices = new APIServices();

class SegmentationAnalysis extends Component {
    constructor(props) {
        super(props);
        this.state = {
            segmentationData: [],
            colors: ["#7852e2", "#b09ee2", "#009688", "#4CAF50", "#d5d5d5", "#92c5de", "#0571b0"],
            isLoading: false
        };
    }

    componentWillReceiveProps = (props) => {
        const { startDate, endDate, regionId, countryId, cityId, commonOD } = props;
        this.setState({ isLoading: true, segmentationData: [] })
        apiServices.getSegmentationBarChart(startDate, endDate, regionId, countryId, cityId, commonOD).then((segmentationData) => {
            this.setState({ isLoading: false })
            if (segmentationData) {
                if (segmentationData.length > 0) {
                    let sortedData = segmentationData.sort((a, b) => (b.Revenue_CY - a.Revenue_LY));
                    sortedData.length = 5;
                    this.setState(
                        { segmentationData: this.normalizeData(sortedData) },
                        () => Chart.drawBarChart(this.state.segmentationData, this.state.colors, 'segmentationB', 'segmentation'));
                }
            }
        });
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

    render() {
        const { colors, isLoading, segmentationData } = this.state;

        return (
            <div className="x_panel tile">
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>Segmentation Analysis</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MutliBarGraphLegends i={true} data={segmentationData} colors={colors} />
                        </div>
                        <li onClick={() => this.props.history.push('/segmentation')}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ?
                    <Spinners /> :
                    segmentationData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '17%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="segmentation"></div>
                            <MutliBarGraphLegends data={segmentationData} colors={colors} />
                        </div>}
            </div>
        )
    }
}

export default SegmentationAnalysis;