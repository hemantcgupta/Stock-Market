import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MultiLineChartLegends from '../../../../Component/MultiLineChartLegends';
import Chart from '../../../../Component/DrawLineChart';
import String from '../../../../Constants/validator';
import Data from '../../../../Constants/validator';
import Spinners from "../../../../spinneranimation";
import '../RouteRevenuePlanning.scss';


const apiServices = new APIServices();

class RevenueBudget extends Component {
    constructor(props) {
        super();
        this.state = {
            revenueData: [],
            isLoading: false,
            colors: ["#009688", "#44c94a", "#49e8a4", "#0571b0", "#92c5de", "#0571b0"],
            selectedTrend: 'Day',
            selectedDelta:'abs'
        };
    }

    componentWillReceiveProps = (props) => {
        this.getGraphData(props)
    }

    getGraphData(props) {
        const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
        const { selectedTrend, selectedDelta } = this.state;
        const group = String.removeQuotes(routeGroup)
        // this.url = `/rps`;
        this.getRouteURL(group, regionId, countryId, routeId);
        this.setState({ isLoading: true, revenueData: [] })
        apiServices.getRevenueBarChart(startDate, endDate, routeGroup, regionId, countryId, routeId, selectedTrend).then((revenueData) => {
            this.setState({ isLoading: false })
            if (revenueData) {
                if (revenueData.length > 0) {
                    let graphData = [...revenueData[0], ...revenueData[1]]
                    if (graphData.length > 0) {
                        this.setState(
                            { revenueData: Data.normalizeLineChartData(graphData, selectedDelta) },
                            () => Chart.drawLineChart(this.state.revenueData, this.state.colors, 'revenueL', 'revenue'));
                    }
                }
            }
        });
    }


    periodChange = (e) => {
        e.stopPropagation();
        this.setState({ selectedDelta: e.target.value }, () => { this.getGraphData(this.props) })
    }

    getRouteURL(routeGroup, regionId, countryId, routeId) {
        let leg = window.localStorage.getItem('LegSelected')
        let flight = window.localStorage.getItem('FlightSelected')

        if (regionId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
        }
        if (countryId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
        }
        if (routeId !== 'Null') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
        }
        if (leg !== null && leg !== 'Null' && leg !== '') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
        }
        if (flight !== null && flight !== 'Null' && flight !== '') {
            this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
        }
    }

    render() {
        const { colors, isLoading, revenueData, selectedTrend } = this.state;
        const uniqueArr = [... new Set(revenueData.map(data => data.Category))]

        return (
            <div className="x_panel tile">
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>Revenue Budget</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <label className='label'>Delta:</label>
                        <select class="select" onChange={(e) => this.periodChange(e)}>
                        <option value='abs'>Absolute</option>
                            <option value='percent'>Percentage</option>
                        </select>
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MultiLineChartLegends i={true} colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>
                        <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
                    </ul>

                </div>
                {isLoading ? <Spinners /> :
                    revenueData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="revenue"></div>
                            <MultiLineChartLegends colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>}
            </div>
        )
    }
}

export default RevenueBudget;
