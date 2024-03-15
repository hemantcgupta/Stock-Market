import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
import Chart from '../../../../Component/DrawBarChart';
import Spinners from "../../../../spinneranimation";
import '../PosRevenuePlanning.scss';


const apiServices = new APIServices();

class RegionPRP extends Component {
    constructor(props) {
        super();
        this.state = {
            performanceData: [],
            colors: ["#009688", "#4CAF50", "#03f574", "#92c5de", "#0571b0", "#92c5de", "#0571b0"],
            isLoading: false,
        };
    }

    componentWillReceiveProps = (props) => {
        const { startDate, endDate, regionId, countryId, cityId } = props;
        this.setState({ isLoading: true, performanceData: [] })
        apiServices.getRegionBudgetChart(startDate, endDate, regionId, countryId, cityId).then((performanceData) => {
            this.setState({ isLoading: false })
            if (performanceData) {
                if (performanceData.Data.length > 0) {
                    let sortedData = performanceData.Data.sort((a, b) => (b.CY_revenue - a.CY_revenue));
                    sortedData.length = 5;
                    this.setState(
                        { performanceData: this.normalizeData(sortedData) },
                        () => Chart.drawBarChart(this.state.performanceData, this.state.colors, 'regionPerformnaceB', 'regionPerformnace'));
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
        const { colors, isLoading, performanceData } = this.state;

        return (
            <div className="x_panel tile">
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>Region</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MutliBarGraphLegends i={true} data={performanceData} colors={colors} />
                        </div>
                        <li onClick={() => this.props.history.push(this.props.url)}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ?
                    <Spinners /> :
                    performanceData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="regionPerformnace"></div>
                            <MutliBarGraphLegends data={performanceData} colors={colors} />
                        </div>}

            </div>
        )
    }
}

export default RegionPRP;
