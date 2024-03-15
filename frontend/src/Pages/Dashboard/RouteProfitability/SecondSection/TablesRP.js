import React, { Component } from "react";
import RouteRegionPerformance from './RouteRegionPerformance';
import AircraftPerformance from './AircraftPerformance';
import TopBottomRoutesRP from './TopBottomRoutesRP';
import '../RouteProfitability.scss';

class TablesRP extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <RouteRegionPerformance
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            typeofCost={this.props.typeofCost}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <AircraftPerformance
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            typeofCost={this.props.typeofCost}
            posDashboard={false}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <TopBottomRoutesRP
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            typeofCost={this.props.typeofCost}
            {...this.props}
          />
        </div>
      </div>
    )
  }
}
export default TablesRP;