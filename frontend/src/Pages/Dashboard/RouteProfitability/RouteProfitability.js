import React, { Component } from "react";
import eventApi from '../../../API/eventApi'
import { Route, NavLink, HashRouter } from "react-router-dom";
import { addDays, subDays } from 'date-fns';
import Indicators from "./RPIndicators";
import TopMenuBar from "../../../Component/TopMenuBar";
import GraphsRP from "../RouteProfitability/FirstSection/GraphsRP";
import TablesRP from "../RouteProfitability/SecondSection/TablesRP"
import cookieStorage from "../../../Constants/cookie-storage";
import Constant from "../../../Constants/validator";
import Access from "../../../Constants/accessValidation";
import ChangePasswordModal from "../../../Component/ChangePasswordModal";
import _ from 'lodash';
import './RouteProfitability.scss'

var date = new Date();

class RouteProfitability extends Component {
    constructor(props) {
        super(props);
        this.state = {
            displayModal: false,
            resetPass: '',
            loading: false,
        }
        this.startDate = '';
        this.endDate = '';
        this.routeGroup = 'Null';
        this.regionSelected = 'Null';
        this.countrySelectedID = 'Null';
        this.routeSelected = 'Null';
        this.middleDashboardRef = React.createRef();
    }

    componentDidMount() {
        if (Access.accessValidation('Dashboard', 'routeProfitability')) {
            // this.getResetPassStatus();
            const token = cookieStorage.getCookie('Authorization')
            this.sendEvent(token);
        } else {
            this.props.history.push('/404')
        }
    }

    sendEvent = (token) => {
        var eventData = {
            event_id: "1",
            description: "User viewed Route Profitability Dashboard Page",
            where_path: "/routeProfitability",
            page_name: "Route Profitability Dashboard Page"
        }
        eventApi.sendEventWithHeader(eventData, token)
    }

    getResetPassStatus() {
        const user = cookieStorage.getCookie('userDetails')
        const userDetails = JSON.parse(user)
        if (userDetails.is_resetpwd === 'True') {
            this.setState({ displayModal: true })
        }
    }

    getSelectedGeographicalDetails = ($event) => {
        this.routeGroup = $event.routeGroup
        this.regionSelected = $event.regionSelected
        this.countrySelectedID = $event.countrySelected
        this.routeSelected = $event.routeSelected
        this.startDate = $event.startDate
        this.endDate = $event.endDate
        this.typeofCost = $event.typeofCost

        // Call to child Component
        this.middleDashboardRef.current.setFilterValues({
            routeGroup: this.routeGroup,
            regionSelected: this.regionSelected === 'All' ? 'Null' : this.regionSelected,
            countrySelectedID: this.countrySelectedID,
            routeSelected: this.routeSelected,
            startDate: this.startDate,
            endDate: this.endDate,
            typeofCost: this.typeofCost
        });
    }

    render() {
        return (
            <div>
                <TopMenuBar
                    dateRange={'currentYear'}
                    dashboardName={'Route Profitability'}
                    routeDash={true}
                    dashboard={true}
                    getSelectedGeographicalDetails={this.getSelectedGeographicalDetails}
                    {...this.props}
                />
                <MiddleDashboard
                    ref={this.middleDashboardRef}
                    {...this.props}
                />
                <ChangePasswordModal displayModal={this.state.displayModal} />
            </div>
        )
    }
}

class MiddleDashboard extends Component {
    constructor() {
        super();
        this.state = {
            routeGroup: null,
            regionId: null,
            routeId: null,
            countryId: null,
            startDate: null,
            endDate: null,
            typeofCost: null
        }
    }

    setFilterValues(args) {
        const { routeGroup, regionSelected, routeSelected, countrySelectedID, startDate, endDate, typeofCost } = args;
        this.setState({
            startDate: startDate,
            endDate: endDate,
            routeGroup: routeGroup,
            regionId: regionSelected,
            countryId: countrySelectedID,
            routeId: routeSelected,
            typeofCost: typeofCost
        })
    }

    render() {
        const { routeGroup, regionId, routeId, countryId, startDate, endDate, typeofCost } = this.state;
        console.log('rahul', endDate)
        return (
            <div className='RouteProfitability'>
                <div className="row tile_count">
                    <Indicators
                        startDate={startDate}
                        endDate={endDate}
                        routeGroup={routeGroup}
                        regionId={regionId}
                        countryId={countryId}
                        routeId={routeId}
                        dashboard={'Route Profitability'}
                        {...this.props}
                    />
                </div>
                <GraphsRP
                    startDate={Constant.getStartEndDateOfMonth(Constant.dateArray(endDate)[1], Constant.dateArray(endDate)[0]).startDate}
                    endDate={Constant.getStartEndDateOfMonth(Constant.dateArray(endDate)[1], Constant.dateArray(endDate)[0]).endDate}
                    routeGroup={routeGroup}
                    regionId={regionId}
                    countryId={countryId}
                    routeId={routeId}
                    {...this.props}
                />
                <div className="row tables">
                    <TablesRP
                        startDate={Constant.addZeroInMonth(startDate)}
                        endDate={Constant.addZeroInMonth(endDate)}
                        routeGroup={routeGroup}
                        regionId={regionId}
                        countryId={countryId}
                        routeId={routeId}
                        typeofCost={typeofCost}
                        {...this.props}
                    />
                </div>
            </div >
        );
    }
}

export default RouteProfitability