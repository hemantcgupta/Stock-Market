import React, { Component } from "react";
import eventApi from '../../../API/eventApi'
import { Route, NavLink, HashRouter } from "react-router-dom";
import { addDays, subDays } from 'date-fns';
import Indicators from "../../../Component/indicators";
import TopMenuBar from "../../../Component/TopMenuBar";
import GraphsRRP from "./FirstSection/GraphsRRP";
import TablesRRP from "./SecondSection/TablesRRP"
import cookieStorage from "../../../Constants/cookie-storage";
import Access from "../../../Constants/accessValidation";
import ChangePasswordModal from "../../../Component/ChangePasswordModal";
import _ from 'lodash';
import './RouteRevenuePlanning.scss'

var date = new Date();

class RouteRevenuePlanning extends Component {
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
        if (Access.accessValidation('Dashboard', 'routeRevenuePlanning')) {
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
            description: "User viewed Route Revenue Planning Dashboard Page",
            where_path: "/routeRevenuePlanning",
            page_name: "Route Revenue Planning Dashboard Page"
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

        // Call to child Component
        this.middleDashboardRef.current.setFilterValues({
            routeGroup: this.routeGroup,
            regionSelected: this.regionSelected === 'All' ? 'Null' : this.regionSelected,
            countrySelectedID: this.countrySelectedID,
            routeSelected: this.routeSelected,
            startDate: this.startDate,
            endDate: this.endDate
        });
    }

    render() {
        return (
            <div>
                <TopMenuBar
                    routeDash={true}
                    dashboard={true}
                    getSelectedGeographicalDetails={this.getSelectedGeographicalDetails} {...this.props}
                    monthRange={'nextYear'}
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
            endDate: null
        }
    }

    setFilterValues(args) {
        const { routeGroup, regionSelected, routeSelected, countrySelectedID, startDate, endDate } = args;
        this.setState({
            startDate: startDate,
            endDate: endDate,
            routeGroup: routeGroup,
            regionId: regionSelected,
            countryId: countrySelectedID,
            routeId: routeSelected
        })
    }

    render() {
        return (
            <div className='RouteRevenuePlanning'>
                <div className="row tile_count">
                    <Indicators
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        routeGroup={this.state.routeGroup}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        routeId={this.state.routeId}
                        dashboard={'Route Revenue Planning'}
                        {...this.props}
                    />
                </div>
                <GraphsRRP
                    startDate={this.state.startDate}
                    endDate={this.state.endDate}
                    routeGroup={this.state.routeGroup}
                    regionId={this.state.regionId}
                    countryId={this.state.countryId}
                    routeId={this.state.routeId}
                    {...this.props}
                />
                <div className="row tables">
                    <TablesRRP
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        routeGroup={this.state.routeGroup}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        routeId={this.state.routeId}
                        {...this.props}
                    />
                </div>
            </div >
        );
    }
}

export default RouteRevenuePlanning