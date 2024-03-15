import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import MultiLineChartLegends from '../../../../Component/MultiLineChartLegends';
import Chart from '../../../../Component/DrawLineChart';
import String from '../../../../Constants/validator';
import Data from '../../../../Constants/validator';
import Spinners from "../../../../spinneranimation";
import '../RouteDashboard.scss';


const apiServices = new APIServices();

class LoadFactor extends Component {
    constructor(props) {
        super();
        this.state = {
            loadFactorData: [],
            colors: ["#0571b0", "#92c5de", '#46b6ed', "#4CAF50", "#d5d5d5", "#92c5de", "#0571b0"],
            isLoading: false,
            selectedTrend: 'Day',
            // selectedDelta: 'abs'
            selectedDelta: 'actual',
        };
    }

    componentWillReceiveProps = (props) => {
        this.getGraphData(props)
    }

    getGraphData(props) {
        const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
        const { selectedTrend, selectedDelta } = this.state;
        const group = String.removeQuotes(routeGroup)
        this.url = `/route?RouteGroup=${group}`;
        this.getRouteURL(group, regionId, countryId, routeId);
        this.setState({ isLoading: true, loadFactorData: [] })

        let endDateArray = endDate.split('-')
        endDateArray.length = 2
        let a = [...endDateArray, ...["01"]]

        apiServices.getLoadFactorBarChart(a.join('-'), endDate, routeGroup, regionId, countryId, routeId, selectedTrend, selectedDelta).then((loadFactorData) => {
            this.setState({ isLoading: false })
            if (loadFactorData) {
                if (loadFactorData[0].length > 0) {
                    let graphData = [...loadFactorData[0], ...loadFactorData[1]]
                    if (graphData.length > 0) {
                        this.setState(
                            { loadFactorData: Data.rectifiedDeltaData(graphData, selectedDelta) },
                            () => Chart.drawLineChart(this.state.loadFactorData, this.state.colors, 'loadFactorL', 'loadFactor', selectedDelta));
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
        const { colors, isLoading, loadFactorData, selectedTrend } = this.state;
        const uniqueArr = [... new Set(loadFactorData.map(data => data.Category))]
        return (
            <div className="x_panel tile" style={{ paddingBottom: '15px' }}>
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>LOAD FACTOR</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <label className='label'>Delta:</label>
                        <select class="select" onChange={(e) => this.periodChange(e)}>
                            <option value='actual'>Actual</option>
                            <option value='pick_up'>Pick-up</option>
                            <option value='pick_up_percent'>Pick-up(%)</option>
                            {/* <option value='abs'>Absolute</option>
                            <option value='percent'>Percentage</option> */}
                        </select>
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            <MultiLineChartLegends i={true} colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>
                        <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ?
                    <Spinners /> :
                    loadFactorData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="loadFactor"></div>
                            <MultiLineChartLegends colors={colors} selectedTrend={selectedTrend} data={uniqueArr} />
                        </div>}
            </div>
        )
    }
}

export default LoadFactor;
